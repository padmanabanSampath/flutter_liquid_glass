import 'dart:math' as math;
import 'dart:ui' show lerpDouble;

import 'package:flutter/gestures.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/common.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/sea_glass_menus.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// Living sea glass: pebbles whose outlines flow and breathe and that merge
/// like drops, a sun that steers the light of every page, and the organic
/// outlines of [LiquidBlob], with a clear dome lens over a tide table. The
/// pebbles can be flat sea glass, a dome or a prism; nothing else in the
/// showcase changes with them.
class TidePoolPage extends StatelessWidget {
  const TidePoolPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ShowcasePageFrame(
      title: 'Tide Pool',
      subtitle: 'Living glass that flows, merges and catches the sun',
      children: [
        SectionHeader('Living glass', trailing: 'Drag one into another'),
        _Pool(),
        SizedBox(height: 26),
        SectionHeader('Dome lens', trailing: 'Slide the lens'),
        _LensCard(),
        SizedBox(height: 26),
        SectionHeader('Outlines', trailing: 'Tap to tumble'),
        _Outlines(),
      ],
    );
  }
}

/// A pool of living pebbles, one blend group, and the sun.
///
/// The pebbles are [LiquidBlob.alive]: their outlines flow and breathe.
/// Drag one into another and the two merge like drops, then pinch apart.
/// The sun lies on the wallpaper, so the glass bends it as it passes; drag
/// it and the light of every page comes from where it is.
class _Pool extends StatefulWidget {
  const _Pool();

  @override
  State<_Pool> createState() => _PoolState();
}

class _PoolState extends State<_Pool> with SingleTickerProviderStateMixin {
  static const _height = 380.0;
  static const _blend = 26.0;
  static const _sunRadius = 24.0;

  /// The width the pebble sizes are drawn for; narrower pools scale them.
  static const _designWidth = 480.0;

  /// Name, color, size, seed and resting place, as a share of the pool, of
  /// each pebble; a `null` color takes the showcase's accent.
  static const _pebbles = <(String?, Color?, double, int, Offset)>[
    ('Ruby', SeaGlassColors.ruby, 96, 1, Offset(.2, .24)),
    ('Clear', SeaGlassColors.clear, 124, 2, Offset(.54, .34)),
    (null, null, 88, 3, Offset(.86, .2)),
    ('Aqua', SeaGlassColors.aqua, 78, 4, Offset(.17, .72)),
    ('Amber', SeaGlassColors.amber, 92, 5, Offset(.48, .78)),
    ('Mist', SeaGlassColors.mist, 84, 6, Offset(.83, .68)),
  ];

  final _time = ValueNotifier<double>(0);
  late final Ticker _ticker = createTicker(
    (elapsed) => _time.value = elapsed.inMicroseconds / 1e6,
  );

  /// Pebble centers in the pool, set on the first layout.
  /// The lens of the pebbles; the rest of the showcase stays flat.
  var _lens = PebbleLens.flat;

  late List<Offset> _centers;
  late Offset _sun;
  double? _laidOutWidth;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final still = MediaQuery.disableAnimationsOf(context);
    if (still && _ticker.isActive) _ticker.stop();
    if (!still && !_ticker.isActive) _ticker.start();
  }

  @override
  void dispose() {
    _ticker.dispose();
    _time.dispose();
    super.dispose();
  }

  void _layOut(Size pool, double lightAngle) {
    if (_laidOutWidth == pool.width) return;
    _laidOutWidth = pool.width;
    _centers = [
      for (final (_, _, _, _, rest) in _pebbles)
        Offset(rest.dx * pool.width, rest.dy * pool.height),
    ];
    // The sun starts where the light already comes from.
    final reach = pool.shortestSide / 2 - _sunRadius - 8;
    _sun =
        pool.center(Offset.zero) +
        Offset(math.sin(lightAngle), -math.cos(lightAngle)) * reach;
  }

  static Offset _keepInside(Offset center, double radius, Size pool) => Offset(
    center.dx.clamp(radius, math.max(radius, pool.width - radius)),
    center.dy.clamp(radius, math.max(radius, pool.height - radius)),
  );

  void _moveSun(Offset delta, Size pool) {
    final sun = _keepInside(_sun + delta, _sunRadius, pool);
    final toSun = sun - pool.center(Offset.zero);
    setState(() => _sun = sun);
    // Clockwise from straight above, as LiquidGlassSettings.lightAngle.
    ShowcaseScope.update(
      context,
      (style) => style.copyWith(lightAngle: math.atan2(toSun.dx, -toSun.dy)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          height: _height,
          child: LayoutBuilder(
            builder: (context, constraints) {
              final pool = Size(constraints.maxWidth, _height);
              _layOut(pool, style.lightAngle);
              final scale = (pool.width / _designWidth).clamp(.62, 1.0);
              final sun = _sun;
              return Stack(
                children: [
                  Positioned.fill(
                    child: CustomPaint(
                      painter: _PoolPainter(ink: style.ink, dark: style.dark),
                    ),
                  ),
                  Positioned(
                    left: sun.dx - _sunRadius * 2,
                    top: sun.dy - _sunRadius * 2,
                    width: _sunRadius * 4,
                    height: _sunRadius * 4,
                    child: Semantics(
                      label: 'Sun',
                      hint: 'Drag to move the light',
                      child: _Drag(
                        onUpdate: (details) => _moveSun(details.delta, pool),
                        child: const CustomPaint(painter: _SunPainter()),
                      ),
                    ),
                  ),
                  Positioned.fill(
                    child: ShowcaseLayer(
                      settings: _lens.apply(
                        style.settings,
                        style.finish.finish,
                      ),
                      child: ValueListenableBuilder(
                        valueListenable: _time,
                        builder: (context, time, _) => LiquidGlassBlendGroup(
                          blend: _blend,
                          child: Stack(
                            children: [
                              for (final (i, pebble) in _pebbles.indexed)
                                _pebble(i, pebble, style, scale, time, pool),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'Drag the sun: the light of every page comes from it. '
          'Light from ${_compass(style.lightAngle)}.',
          textAlign: TextAlign.center,
          style: TextStyle(
            color: style.ink.withValues(alpha: .72),
            fontSize: 13,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 14),
        Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 300),
            child: ShowcaseLayer(
              child: GlassSegmented(
                labels: [for (final lens in PebbleLens.values) lens.label],
                selected: _lens.index,
                ink: style.ink,
                body: style.roleBody(SlabRole.positive),
                onSelected: (i) => setState(() => _lens = PebbleLens.values[i]),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _pebble(
    int index,
    (String?, Color?, double, int, Offset) pebble,
    ShowcaseStyle style,
    double scale,
    double time,
    Size pool,
  ) {
    final (name, color, diameter, seed, _) = pebble;
    final size = diameter * scale;
    final body = style.bodyOf(color ?? style.accent);
    final center = _centers[index];
    return Positioned(
      left: center.dx - size / 2,
      top: center.dy - size / 2,
      width: size,
      height: size,
      child: Semantics(
        label: '${name ?? style.accentName} pebble',
        hint: 'Drag into another pebble to merge',
        child: _Drag(
          onUpdate: (details) => setState(
            () => _centers[index] = _keepInside(
              // The current center: several moves can arrive in one frame.
              _centers[index] + details.delta,
              size / 2,
              pool,
            ),
          ),
          child: LiquidStretch.seaGlass(
            child: LiquidGlass.grouped(
              shape: LiquidBlob.alive(time: time, seed: seed),
              appearance: LiquidGlassAppearance.seaGlass(color: body),
              shadows: showcaseShadows(context, body),
              child: Center(
                child: Text(
                  name ?? style.accentName,
                  style: TextStyle(
                    color: inkOn(body),
                    shadows: haloFor(inkOn(body)),
                    fontSize: 13 * scale.clamp(.85, 1.0),
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// The compass point the light comes from.
  static String _compass(double angle) {
    const points = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    final step = (angle / (math.pi / 4)).round() % points.length;
    return points[step];
  }
}

/// The basin of the tide pool: still water, sand and rings of ripples for
/// the glass to bend.
class _PoolPainter extends CustomPainter {
  const _PoolPainter({required this.ink, required this.dark});

  final Color ink;
  final bool dark;

  static const _ripples = [Offset(.3, .5), Offset(.72, .42), Offset(.5, .9)];

  @override
  void paint(Canvas canvas, Size size) {
    final basin = RRect.fromRectAndRadius(
      Offset.zero & size,
      const Radius.circular(36),
    );
    canvas
      ..drawRRect(
        basin,
        Paint()
          ..shader = LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: dark
                ? const [Color(0x2A9FE3E0), Color(0x0F0E4F5A)]
                : const [Color(0x4DFFFFFF), Color(0x1FFFFFFF)],
          ).createShader(Offset.zero & size),
      )
      ..save()
      ..clipRRect(basin);
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    for (final share in _ripples) {
      final center = Offset(share.dx * size.width, share.dy * size.height);
      for (var i = 0; i < 6; i++) {
        ring.color = ink.withValues(alpha: .16 - i * .022);
        canvas.drawCircle(center, 22.0 + i * 24, ring);
      }
    }
    final random = math.Random(7);
    final grain = Paint();
    for (var i = 0; i < 320; i++) {
      grain.color = ink.withValues(alpha: .08 + random.nextDouble() * .14);
      canvas.drawCircle(
        Offset(
          random.nextDouble() * size.width,
          random.nextDouble() * size.height,
        ),
        .6 + random.nextDouble() * 1.1,
        grain,
      );
    }
    canvas
      ..restore()
      ..drawRRect(
        basin,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1
          ..color = ink.withValues(alpha: .16),
      );
  }

  @override
  bool shouldRepaint(_PoolPainter oldDelegate) =>
      oldDelegate.ink != ink || oldDelegate.dark != dark;
}

/// A warm sun with a soft halo, half as wide as its box.
class _SunPainter extends CustomPainter {
  const _SunPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final halo = size.shortestSide / 2;
    final core = halo / 2;
    canvas
      ..drawCircle(
        center,
        halo,
        Paint()
          ..shader = RadialGradient(
            colors: [
              const Color(0xFFFFC94D).withValues(alpha: .7),
              const Color(0x00FFC94D),
            ],
          ).createShader(Rect.fromCircle(center: center, radius: halo)),
      )
      ..drawCircle(
        center,
        core,
        Paint()
          ..shader = const RadialGradient(
            colors: [Color(0xFFFFF1C2), Color(0xFFFFB238)],
          ).createShader(Rect.fromCircle(center: center, radius: core)),
      );
  }

  @override
  bool shouldRepaint(_SunPainter oldDelegate) => false;
}

/// A tide table under a clear dome of sea glass.
///
/// The lens is a pebble of clear glass with a prism's dispersion in a layer
/// of its own just around it, so moving it moves the layer and never draws
/// its geometry again. Its dome bows the fine print and the tide curve and
/// fringes their edges with color.
class _LensCard extends StatefulWidget {
  const _LensCard();

  @override
  State<_LensCard> createState() => _LensCardState();
}

class _LensCardState extends State<_LensCard> {
  static const _height = 214.0;
  static const _lens = 118.0;

  Offset? _center;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final settings = LiquidGlassSettings.seaGlass(
      finish: const SeaGlassFinish.clear(),
      frost: 0,
      dispersion: -.25,
    ).copyWith(lightAngle: style.lightAngle);
    final body = SeaGlassColors.clear.withValues(alpha: .3);
    return SizedBox(
      height: _height,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final width = constraints.maxWidth;
          final center = _center ??= Offset(width * .7, _height / 2);
          return Stack(
            children: [
              const Positioned.fill(
                child: ShowcaseLayer(
                  child: GlassCard(
                    padding: EdgeInsets.fromLTRB(18, 16, 18, 14),
                    child: _TideTable(),
                  ),
                ),
              ),
              Positioned(
                left: center.dx - _lens / 2,
                top: center.dy - _lens / 2,
                width: _lens,
                height: _lens,
                child: Semantics(
                  label: 'Lens',
                  hint: 'Drag over the tide table',
                  child: _Drag(
                    // The current center: several moves can arrive in one
                    // frame.
                    onUpdate: (details) => setState(
                      () => _center = Offset(
                        (_center!.dx + details.delta.dx).clamp(
                          _lens / 2,
                          math.max(_lens / 2, width - _lens / 2),
                        ),
                        (_center!.dy + details.delta.dy).clamp(
                          _lens / 2,
                          _height - _lens / 2,
                        ),
                      ),
                    ),
                    child: LiquidGlassLayer(
                      settings: settings,
                      child: LiquidGlass(
                        shape: LiquidBlob.pebble(seed: 9, irregularity: .4),
                        appearance: LiquidGlassAppearance.seaGlass(color: body),
                        shadows: SeaGlassFinish.shadows(
                          SeaGlassColors.cornflower,
                          elevation: 8,
                        ),
                        child: const SizedBox.expand(),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _TideTable extends StatelessWidget {
  const _TideTable();

  static const _tides = [
    ('High', '05:42', '1.8 m'),
    ('Low', '11:58', '0.3 m'),
    ('High', '18:21', '1.6 m'),
    ('Low', '23:47', '0.4 m'),
  ];

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final ink = style.ink;
    final fine = TextStyle(
      color: ink.withValues(alpha: .72),
      fontSize: 11,
      fontWeight: FontWeight.w500,
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Text(
              'Tides · Fort Kochi',
              style: TextStyle(
                color: ink,
                fontSize: 15,
                fontWeight: FontWeight.w700,
              ),
            ),
            const Spacer(),
            Text('Spring tide · Moon 94%', style: fine),
          ],
        ),
        const SizedBox(height: 6),
        Expanded(
          child: CustomPaint(
            painter: _TideCurvePainter(ink: ink, accent: style.accent),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            for (final (kind, time, height) in _tides)
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(kind.toUpperCase(), style: fine),
                    Text(
                      time,
                      style: TextStyle(
                        color: ink,
                        fontSize: 17,
                        fontWeight: FontWeight.w600,
                        fontFeatures: const [FontFeature.tabularFigures()],
                      ),
                    ),
                    Text(height, style: fine),
                  ],
                ),
              ),
          ],
        ),
      ],
    );
  }
}

/// A day of tide over a fine grid of hours, for the lens to bend.
class _TideCurvePainter extends CustomPainter {
  const _TideCurvePainter({required this.ink, required this.accent});

  final Color ink;
  final Color accent;

  @override
  void paint(Canvas canvas, Size size) {
    final grid = Paint()
      ..color = ink.withValues(alpha: .12)
      ..strokeWidth = 1;
    for (var hour = 0; hour <= 24; hour++) {
      final x = size.width * hour / 24;
      canvas.drawLine(
        Offset(x, hour.isEven ? 0 : size.height * .4),
        Offset(x, size.height),
        grid,
      );
    }
    for (var row = 1; row < 4; row++) {
      final y = size.height * row / 4;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }
    // Two tides a day, 12.4 hours apart, the first high at 5:42.
    double level(double hour) =>
        .5 + .38 * math.cos(2 * math.pi * (hour - 5.7) / 12.42);
    final curve = Path();
    final fill = Path()..moveTo(0, size.height);
    for (var i = 0; i <= 96; i++) {
      final hour = 24 * i / 96;
      final point = Offset(
        size.width * hour / 24,
        size.height * (1 - level(hour)),
      );
      if (i == 0) {
        curve.moveTo(point.dx, point.dy);
      } else {
        curve.lineTo(point.dx, point.dy);
      }
      fill.lineTo(point.dx, point.dy);
    }
    fill
      ..lineTo(size.width, size.height)
      ..close();
    final solid = accent.withValues(alpha: 1);
    canvas
      ..drawPath(
        fill,
        Paint()
          ..shader = LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [solid.withValues(alpha: .35), solid.withValues(alpha: 0)],
          ).createShader(Offset.zero & size),
      )
      ..drawPath(
        curve,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = solid,
      );
  }

  @override
  bool shouldRepaint(_TideCurvePainter oldDelegate) =>
      oldDelegate.ink != ink || oldDelegate.accent != accent;
}

/// The outlines a [LiquidBlob] can take, each in a new sea glass color.
/// Tapping one tumbles it: the pebble morphs into another, the others turn.
class _Outlines extends StatelessWidget {
  const _Outlines();

  /// A tumbled pebble at [turn]: it morphs from one pebble into the next as
  /// [turn] passes each whole number.
  static LiquidShape _pebble(double turn) {
    final seed = turn.floor();
    final from = LiquidBlob.pebble(seed: seed).waves;
    final to = LiquidBlob.pebble(seed: seed + 1).waves;
    final t = turn - seed;
    return LiquidBlob(
      waves: [
        for (var i = 0; i < from.length; i++)
          LiquidWave(
            count: from[i].count,
            amplitude: lerpDouble(from[i].amplitude, to[i].amplitude, t)!,
            phase: lerpDouble(from[i].phase, to[i].phase, t)!,
          ),
      ],
    );
  }

  /// [waves] turned by [angle], in radians: each wave of `count` crests
  /// shifts its phase by `count` times the angle.
  static LiquidShape _turned(List<(int, double)> waves, double angle) =>
      LiquidBlob(
        waves: [
          for (final (count, amplitude) in waves)
            LiquidWave(
              count: count,
              amplitude: amplitude,
              phase: -count * angle,
            ),
        ],
      );

  @override
  Widget build(BuildContext context) {
    final tiles = <Widget>[
      const _OutlineTile(
        name: 'Pebble',
        color: SeaGlassColors.ruby,
        outline: _pebble,
      ),
      _OutlineTile(
        name: 'Cookie',
        color: SeaGlassColors.amber,
        // One tap turns it by one of its ten lobes.
        outline: (turn) => _turned(const [(10, .05)], turn * 2 * math.pi / 10),
      ),
      _OutlineTile(
        name: 'Flower',
        color: SeaGlassColors.lavender,
        outline: (turn) => _turned(const [(6, .13)], turn * 2 * math.pi / 6),
      ),
      _OutlineTile(
        name: 'Shell',
        color: SeaGlassColors.mist,
        outline: (turn) =>
            _turned(const [(2, .12), (9, .035)], turn * math.pi / 2),
      ),
    ];
    return ShowcaseLayer(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (var i = 0; i < tiles.length; i++) ...[
            if (i > 0) const SizedBox(width: 14),
            Expanded(child: tiles[i]),
          ],
        ],
      ),
    );
  }
}

class _OutlineTile extends StatefulWidget {
  const _OutlineTile({
    required this.name,
    required this.color,
    required this.outline,
  });

  final String name;
  final Color color;

  /// The outline after `turn` taps, animating in between.
  final LiquidShape Function(double turn) outline;

  @override
  State<_OutlineTile> createState() => _OutlineTileState();
}

class _OutlineTileState extends State<_OutlineTile> {
  var _turns = 0;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final body = style.bodyOf(widget.color);
    return Semantics(
      button: true,
      label: '${widget.name} outline',
      hint: 'Tap to tumble',
      excludeSemantics: true,
      child: GestureDetector(
        onTap: () => setState(() => _turns++),
        child: Column(
          children: [
            AspectRatio(
              aspectRatio: 1,
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: _turns.toDouble()),
                duration: const Duration(milliseconds: 900),
                curve: Curves.easeOutBack,
                builder: (context, turn, _) => LiquidStretch.seaGlass(
                  child: LiquidGlass(
                    shape: widget.outline(turn),
                    appearance: LiquidGlassAppearance.seaGlass(color: body),
                    shadows: showcaseShadows(context, body, elevation: 5),
                    child: const SizedBox.expand(),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              widget.name,
              style: TextStyle(
                color: style.ink,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// A drag that claims the gesture at the touch slop, ahead of the pager
/// and the page's scroll view, so pebbles, the sun and the lens move in any
/// direction instead of turning or scrolling the page.
class _Drag extends StatelessWidget {
  const _Drag({required this.onUpdate, required this.child});

  final GestureDragUpdateCallback onUpdate;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    // A pan waits for twice the touch slop in any direction, the pager for
    // the touch slop along its axis. Android sets the slop (about 8 pt), so
    // start from the device's and give the pan less than half of it.
    final slop =
        MediaQuery.maybeGestureSettingsOf(context)?.touchSlop ?? kTouchSlop;
    return RawGestureDetector(
      // Glass with nothing in it takes no hit by itself, so the whole
      // area answers; an empty lens would otherwise not move.
      behavior: HitTestBehavior.opaque,
      gestures: {
        PanGestureRecognizer:
            GestureRecognizerFactoryWithHandlers<PanGestureRecognizer>(
              PanGestureRecognizer.new,
              (pan) => pan
                ..gestureSettings = DeviceGestureSettings(touchSlop: slop * .4)
                ..onUpdate = onUpdate,
            ),
      },
      child: child,
    );
  }
}
