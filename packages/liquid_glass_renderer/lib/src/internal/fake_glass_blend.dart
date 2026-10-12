import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/rendering.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer/src/internal/rounded_superellipse_parameters.dart';
import 'package:meta/meta.dart';

/// Most members one FakeGlass surface blends with, itself included.
///
/// Matches `kFakeGroupMembers` in fake_glass_surface.frag.
@internal
const int fakeGlassGroupMembers = 4;

/// A shape of a FakeGlass blend group.
///
/// FakeGlass draws each shape on its own, so a blend group is drawn as the
/// smooth union of its members by each member's surface: every member
/// evaluates the union with its nearest neighbors and paints the pixels it
/// is nearest to. This is the CPU side of that union, in the layer's logical
/// pixels: it finds the neighbors and traces the bridges between members,
/// which the backdrop must cover too. Its signed distances follow
/// fake_glass_shape.glsl.
@internal
final class FakeGlassBlendMember {
  /// Creates a member [index]th in widget order, of [size] and placed in the
  /// layer by [shapeToLayer].
  FakeGlassBlendMember({
    required this.index,
    required this.shape,
    required this.size,
    required Matrix4 shapeToLayer,
    required this.appearance,
  }) : _toLayer = _Affine.fromMatrix(shapeToLayer) {
    _fromLayer = _toLayer.inverted();
    distanceScale = _toLayer.minimumScale();
    bounds = MatrixUtils.transformRect(shapeToLayer, Offset.zero & size);
    type = switch (shape) {
      LiquidOval() => 0.0,
      LiquidRoundedRectangle() => 1.0,
      LiquidRoundedSuperellipse() => 2.0,
    };
    radius = switch (shape) {
      LiquidOval() => 0.0,
      LiquidRoundedRectangle(:final borderRadius) => borderRadius,
      LiquidRoundedSuperellipse(:final borderRadius) => borderRadius,
    };
    superellipse = shape is LiquidRoundedSuperellipse
        ? roundedSuperellipseParameters(size, radius)
        : const [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  }

  /// Position in widget order, which is the order of the union.
  final int index;
  final LiquidShape shape;
  final Size size;
  final LiquidGlassAppearance appearance;

  final _Affine _toLayer;
  late final _Affine _fromLayer;

  /// Scale from the member's local distances to layer distances; the
  /// smallest singular value, so they are never overestimated.
  late final double distanceScale;

  /// The member's box in the layer.
  late final Rect bounds;

  /// Shape type as fake_glass_shape.glsl numbers it, and corner radius.
  late final double type;
  late final double radius;

  /// Rounded superellipse parameters (see roundedSuperellipseParameters).
  late final List<double> superellipse;

  /// Signed distance in layer pixels and the outward normal in the layer's
  /// frame at [point], a layer position.
  (double, Offset) sample(Offset point) {
    final local = _fromLayer.apply(point) - size.center(Offset.zero);
    final distance = _shapeDistance(local) * distanceScale;
    final n = _shapeNormal(local);
    final normal = _fromLayer.transposeApply(n);
    final length = normal.distance;
    return (distance, length > 1e-9 ? normal / length : Offset.zero);
  }

  /// The affine map from this member's centered local frame to [other]'s,
  /// as six floats: the 2x2 matrix row by row, then the offset.
  List<double> frameTo(FakeGlassBlendMember other) {
    final map = other._fromLayer.multiply(_toLayer);
    final offset =
        map.apply(size.center(Offset.zero)) - other.size.center(Offset.zero);
    return [map.a, map.c, map.b, map.d, offset.dx, offset.dy];
  }

  double _shapeDistance(Offset p) {
    final halfSize = Offset(size.width / 2, size.height / 2);
    if (type < 0.5) return _sdEllipse(p, halfSize);
    final box = _sdRoundedBox(p, halfSize, radius);
    if (type < 1.5 || box < -6) return box;
    final superellipse = _sdRoundedSuperellipse(
      p,
      halfSize,
      math.min(radius, math.min(halfSize.dx, halfSize.dy)),
    );
    final t = ((-box - 4) / 2).clamp(0.0, 1.0);
    return superellipse + (box - superellipse) * t * t * (3 - 2 * t);
  }

  Offset _shapeNormal(Offset p) {
    final halfSize = Offset(size.width / 2, size.height / 2);
    if (type < 0.5) {
      final rx = math.max(halfSize.dx, 0.001);
      final ry = math.max(halfSize.dy, 0.001);
      return _normalize(
        Offset(p.dx / (rx * rx) + 0.00001, p.dy / (ry * ry) + 0.00001),
      );
    }
    final r = radius.clamp(0.0, math.min(halfSize.dx, halfSize.dy));
    final qx = p.dx.abs() - halfSize.dx + r;
    final qy = p.dy.abs() - halfSize.dy + r;
    final cx = math.max<double>(qx, 0);
    final cy = math.max<double>(qy, 0);
    if (cx * cx + cy * cy > 0.0001) {
      return _normalize(Offset(cx, cy)).scale(_sign(p.dx), _sign(p.dy));
    }
    return qx > qy ? Offset(_sign(p.dx), 0) : Offset(0, _sign(p.dy));
  }

  double _sdRoundedSuperellipse(Offset p, Offset halfSize, double radius) {
    if (radius <= 1e-4) {
      return math.max(p.dx.abs() - halfSize.dx, p.dy.abs() - halfSize.dy);
    }
    final s = superellipse;
    final localX = p.dx.abs();
    final localY = p.dy.abs();
    final c = s[8] - s[9];
    double octantX;
    double octantY;
    double degree;
    double span;
    double axis;
    double circleRadius;
    double centerX;
    double centerY;
    if (localY + c > localX) {
      octantX = localX;
      octantY = localY + c;
      degree = s[0];
      axis = s[8];
      span = s[2];
      centerX = s[4];
      centerY = s[5];
      circleRadius = s[10];
    } else {
      octantX = localY;
      octantY = localX - c;
      degree = s[1];
      axis = s[9];
      span = s[3];
      centerX = s[6];
      centerY = s[7];
      circleRadius = s[11];
    }
    final relativeX = octantX - centerX;
    final relativeY = octantY - centerY;
    var deltaTheta = math.atan2(relativeY, relativeX) - math.pi / 4;
    deltaTheta = (deltaTheta + math.pi) % (2 * math.pi) - math.pi;
    if (deltaTheta.abs() < span.abs()) {
      return math.sqrt(relativeX * relativeX + relativeY * relativeY) -
          circleRadius;
    }
    if (degree < 2) {
      return math.max(octantX.abs() - axis, octantY.abs() - axis);
    }
    final scale = math.max(axis, 1e-4);
    return _sdSuperellipseArc(octantX / scale, octantY / scale, degree) * axis;
  }

  static double _sdSuperellipseArc(double x, double y, double n) {
    final px = x.abs() + 1e-6;
    final py = y.abs() + 1e-6;
    final s = math.pow(px, n) + math.pow(py, n);
    final f = math.pow(s, 1 / n) - 1;
    final common = math.pow(s, 1 / n - 1);
    final gx = math.pow(px, n - 1) * common;
    final gy = math.pow(py, n - 1) * common;
    return f / math.max(math.sqrt(gx * gx + gy * gy), 1e-3);
  }

  static double _sdRoundedBox(Offset p, Offset halfSize, double radius) {
    final r = radius.clamp(0.0, math.min(halfSize.dx, halfSize.dy));
    final qx = p.dx.abs() - halfSize.dx + r;
    final qy = p.dy.abs() - halfSize.dy + r;
    final ox = math.max(qx, 0);
    final oy = math.max(qy, 0);
    return math.sqrt(ox * ox + oy * oy) + math.min(math.max(qx, qy), 0.0) - r;
  }

  static double _sdEllipse(Offset p, Offset radius) {
    final rx = math.max(radius.dx, 0.001);
    final ry = math.max(radius.dy, 0.001);
    final k0 = math.sqrt(
      (p.dx / rx) * (p.dx / rx) + (p.dy / ry) * (p.dy / ry),
    );
    final k1 = math.sqrt(
      (p.dx / (rx * rx)) * (p.dx / (rx * rx)) +
          (p.dy / (ry * ry)) * (p.dy / (ry * ry)),
    );
    return k0 * (k0 - 1) / math.max(k1, 0.001);
  }

  static double _sign(double value) => value < 0 ? -1 : (value > 0 ? 1 : 0);

  static Offset _normalize(Offset value) {
    final length = value.distance;
    return length > 0 ? value / length : value;
  }
}

/// The smooth union of [members], in widget order, at a layer [point]:
/// its signed distance and normal, as the FakeGlass surface computes it and
/// as RealGlass blends a group (see `smoothUnionSample` in sdf.glsl).
@internal
(double, Offset) fakeGlassUnion(
  List<FakeGlassBlendMember> members,
  Offset point,
  double blend,
) => _union([for (final member in members) member.sample(point)], blend);

(double, Offset) _union(List<(double, Offset)> samples, double blend) {
  var (distance, normal) = samples.first;
  for (final (other, otherNormal) in samples.skip(1)) {
    final radius = blend * math.min(0.5 * (normal - otherNormal).distance, 1);
    final e = math.max(radius - (distance - other).abs(), 0);
    final weight = (0.5 + (other - distance) / (2 * math.max(radius, 1e-4)))
        .clamp(0.0, 1.0);
    distance =
        math.min(distance, other) - e * e * 0.25 / math.max(radius, 1e-4);
    normal = otherNormal + (normal - otherNormal) * weight;
  }
  return (distance, normal);
}

/// The members a member's surface blends with: itself and its nearest
/// neighbors, at most [fakeGlassGroupMembers] in all, in widget order.
///
/// Neighbors count when their boxes come within [reach] of the member's,
/// so a pixel it paints never lies nearer to a member it left out.
@internal
List<FakeGlassBlendMember> fakeGlassNeighbors(
  FakeGlassBlendMember member,
  List<FakeGlassBlendMember> group,
  double reach,
) {
  final candidates = <(double, FakeGlassBlendMember)>[];
  for (final other in group) {
    if (identical(other, member)) continue;
    final distance = _boxDistance(member.bounds, other.bounds);
    if (distance <= reach) candidates.add((distance, other));
  }
  if (candidates.isEmpty) return const [];
  candidates.sort((a, b) => a.$1.compareTo(b.$1));
  return [
    member,
    for (final (_, other) in candidates.take(fakeGlassGroupMembers - 1)) other,
  ]..sort((a, b) => a.index.compareTo(b.index));
}

double _boxDistance(Rect a, Rect b) {
  final dx = math.max(0, math.max(a.left - b.right, b.left - a.right));
  final dy = math.max(0, math.max(a.top - b.bottom, b.top - a.bottom));
  return math.sqrt(dx * dx + dy * dy);
}

/// Spacing, in layer pixels, of the grid the bridges are traced on. Linear
/// interpolation along its edges keeps the outline within a fraction of a
/// pixel on these smooth curves.
const double _bridgeGridStep = 2;

/// How far, in layer pixels, a bridge reaches into the members it joins, so
/// its antialiased edge and theirs leave no seam.
const double _bridgeOverlap = 1;

/// The material that joins [group]'s members: for every pair close enough
/// to bridge, the part of their [blend] union between them, outlined as
/// closed polygons and split between the two by which is nearer, so each
/// half can take its own member's backdrop. Keyed by member.
@internal
Map<FakeGlassBlendMember, Path> fakeGlassBridges(
  List<FakeGlassBlendMember> group,
  double blend,
) {
  final bridges = <FakeGlassBlendMember, Path>{};
  if (blend <= 0) return bridges;
  for (var i = 0; i < group.length; i++) {
    for (var j = i + 1; j < group.length; j++) {
      final a = group[i].bounds;
      final b = group[j].bounds;
      // The union lies at most a quarter radius outside the nearer member,
      // so a bridge forms only across a gap below half the radius.
      if (_boxDistance(a, b) >= blend / 2) continue;
      final region = a.inflate(blend).intersect(b.inflate(blend));
      if (region.isEmpty) continue;
      final members = [
        for (final member in group)
          if (member.bounds.inflate(blend).overlaps(region)) member,
      ];
      for (final owner in [group[i], group[j]]) {
        // Inside the union, outside the members but for the overlap, and
        // nearer the owner than any other member.
        double field(Offset point) {
          final samples = [for (final member in members) member.sample(point)];
          final union = _union(samples, blend).$1;
          var own = 0.0;
          var nearest = double.infinity;
          var others = double.infinity;
          for (final (index, member) in members.indexed) {
            final distance = samples[index].$1;
            nearest = math.min(nearest, distance);
            if (identical(member, owner)) {
              own = distance;
            } else {
              others = math.min(others, distance);
            }
          }
          return math.max(
            union,
            math.max(-nearest - _bridgeOverlap, own - others),
          );
        }

        for (final polygon in _traceInside(region, field)) {
          (bridges[owner] ??= Path()).addPolygon(polygon, true);
        }
      }
    }
  }
  return bridges;
}

/// Closed outlines of where [field] is negative within [region], traced by
/// marching squares. The grid's border counts as outside, so every outline
/// closes.
List<List<Offset>> _traceInside(
  Rect region,
  double Function(Offset point) field,
) {
  final columns = math.max(2, (region.width / _bridgeGridStep).ceil() + 1);
  final rows = math.max(2, (region.height / _bridgeGridStep).ceil() + 1);
  final stepX = region.width / (columns - 1);
  final stepY = region.height / (rows - 1);
  final values = Float64List(columns * rows);
  Offset point(int column, int row) =>
      Offset(region.left + column * stepX, region.top + row * stepY);
  for (var row = 0; row < rows; row++) {
    for (var column = 0; column < columns; column++) {
      final border =
          row == 0 || column == 0 || row == rows - 1 || column == columns - 1;
      final value = field(point(column, row));
      // Exactly zero would sit on the outline; nudge it outside.
      values[row * columns + column] = border
          ? math.max(value, 1e-3)
          : (value == 0 ? 1e-6 : value);
    }
  }
  double at(int column, int row) => values[row * columns + column];

  // Edge keys: horizontal edges even, vertical ones odd.
  int horizontal(int column, int row) => 2 * (row * columns + column);
  int vertical(int column, int row) => 2 * (row * columns + column) + 1;
  final points = <int, Offset>{};
  Offset crossing(int key) => points.putIfAbsent(key, () {
    final index = key >> 1;
    final column = index % columns;
    final row = index ~/ columns;
    final a = point(column, row);
    final b = key.isEven ? point(column + 1, row) : point(column, row + 1);
    final fa = at(column, row);
    final fb = key.isEven ? at(column + 1, row) : at(column, row + 1);
    final t = (fa / (fa - fb)).clamp(0.0, 1.0);
    return a + (b - a) * t;
  });

  // Segments run with the inside on their right, so chaining them traces
  // every outline the same way round.
  final next = <int, int>{};
  for (var row = 0; row < rows - 1; row++) {
    for (var column = 0; column < columns - 1; column++) {
      final f0 = at(column, row);
      final f1 = at(column + 1, row);
      final f2 = at(column + 1, row + 1);
      final f3 = at(column, row + 1);
      final top = horizontal(column, row);
      final right = vertical(column + 1, row);
      final bottom = horizontal(column, row + 1);
      final left = vertical(column, row);
      final mask =
          (f0 < 0 ? 1 : 0) |
          (f1 < 0 ? 2 : 0) |
          (f2 < 0 ? 4 : 0) |
          (f3 < 0 ? 8 : 0);
      void link(int from, int to) => next[from] = to;
      switch (mask) {
        case 1:
          link(top, left);
        case 2:
          link(right, top);
        case 3:
          link(right, left);
        case 4:
          link(bottom, right);
        case 5:
          if ((f0 + f1 + f2 + f3) / 4 < 0) {
            link(top, right);
            link(bottom, left);
          } else {
            link(top, left);
            link(bottom, right);
          }
        case 6:
          link(bottom, top);
        case 7:
          link(bottom, left);
        case 8:
          link(left, bottom);
        case 9:
          link(top, bottom);
        case 10:
          if ((f0 + f1 + f2 + f3) / 4 < 0) {
            link(left, top);
            link(right, bottom);
          } else {
            link(right, top);
            link(left, bottom);
          }
        case 11:
          link(right, bottom);
        case 12:
          link(left, right);
        case 13:
          link(top, right);
        case 14:
          link(left, top);
      }
    }
  }

  final polygons = <List<Offset>>[];
  while (next.isNotEmpty) {
    final start = next.keys.first;
    final polygon = <Offset>[];
    var key = start;
    while (true) {
      polygon.add(crossing(key));
      final following = next.remove(key);
      if (following == null || following == start) break;
      key = following;
    }
    if (polygon.length >= 3) polygons.add(polygon);
  }
  return polygons;
}

/// A 2D affine map: x' = a x + c y + tx, y' = b x + d y + ty.
final class _Affine {
  const _Affine(this.a, this.b, this.c, this.d, this.tx, this.ty);

  factory _Affine.fromMatrix(Matrix4 matrix) {
    final m = matrix.storage;
    return _Affine(m[0], m[1], m[4], m[5], m[12], m[13]);
  }

  final double a;
  final double b;
  final double c;
  final double d;
  final double tx;
  final double ty;

  Offset apply(Offset p) =>
      Offset(a * p.dx + c * p.dy + tx, b * p.dx + d * p.dy + ty);

  /// The transpose of the linear part applied to [v]: how a gradient in
  /// the target frame maps back to the source frame.
  Offset transposeApply(Offset v) =>
      Offset(a * v.dx + b * v.dy, c * v.dx + d * v.dy);

  _Affine inverted() {
    final determinant = a * d - b * c;
    if (determinant.abs() < 1e-12) return const _Affine(1, 0, 0, 1, 0, 0);
    final ia = d / determinant;
    final ib = -b / determinant;
    final ic = -c / determinant;
    final id = a / determinant;
    return _Affine(
      ia,
      ib,
      ic,
      id,
      -(ia * tx + ic * ty),
      -(ib * tx + id * ty),
    );
  }

  /// This map after [first].
  _Affine multiply(_Affine first) => _Affine(
    a * first.a + c * first.b,
    b * first.a + d * first.b,
    a * first.c + c * first.d,
    b * first.c + d * first.d,
    a * first.tx + c * first.ty + tx,
    b * first.tx + d * first.ty + ty,
  );

  /// The smallest singular value of the linear part.
  double minimumScale() {
    final trace = a * a + b * b + c * c + d * d;
    final determinant = a * d - b * c;
    final discriminant = math.max(
      0,
      trace * trace - 4 * determinant * determinant,
    );
    return math.sqrt(math.max(0.0, (trace - math.sqrt(discriminant)) / 2));
  }
}
