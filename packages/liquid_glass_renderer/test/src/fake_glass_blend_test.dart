import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer/src/internal/fake_glass_blend.dart';

FakeGlassBlendMember _tile(int index, double left, {double top = 0}) =>
    FakeGlassBlendMember(
      index: index,
      shape: const LiquidRoundedSuperellipse(borderRadius: 20),
      size: const Size(80, 50),
      shapeToLayer: Matrix4.translationValues(left, top, 0),
      appearance: const LiquidGlassAppearance(),
    );

void main() {
  group('FakeGlass blend groups', () {
    test('bridge a gap below half the blend radius, split by member', () {
      final group = [_tile(0, 0), _tile(1, 84)];
      final bridges = fakeGlassBridges(group, 24);
      expect(bridges.keys, unorderedEquals(group));
      // The middle of the gap is covered, each side by its own member.
      final union = Path()
        ..addPath(bridges[group[0]]!, Offset.zero)
        ..addPath(bridges[group[1]]!, Offset.zero);
      expect(union.contains(const Offset(82, 25)), isTrue);
      expect(bridges[group[0]]!.contains(const Offset(81, 25)), isTrue);
      expect(bridges[group[1]]!.contains(const Offset(83, 25)), isTrue);
      expect(fakeGlassUnion(group, const Offset(82, 25), 24).$1, lessThan(0));

      // Twelve apart, the union stays apart.
      expect(fakeGlassBridges([_tile(0, 0), _tile(1, 92)], 24), isEmpty);
      expect(fakeGlassBridges(group, 0), isEmpty);
    });

    test('a surface blends with its nearest neighbors in widget order', () {
      final group = [
        for (var i = 0; i < 6; i++) _tile(i, i * 90.0),
      ];
      final neighbors = fakeGlassNeighbors(group[2], group, 30);
      expect(neighbors, [group[1], group[2], group[3]]);
      expect(fakeGlassNeighbors(group[2], group, 5), isEmpty);
      // At most four members, the nearest ones.
      final crowd = [
        for (var i = 0; i < 6; i++) _tile(i, i * 10.0),
      ];
      expect(
        fakeGlassNeighbors(crowd[0], crowd, 200),
        hasLength(fakeGlassGroupMembers),
      );
      expect(fakeGlassNeighbors(crowd[0], crowd, 200).first, crowd[0]);
    });

    Future<ByteData> capture(WidgetTester tester, Widget scene) async {
      final key = GlobalKey();
      await tester.pumpWidget(
        Directionality(
          textDirection: TextDirection.ltr,
          child: RepaintBoundary(
            key: key,
            child: ColoredBox(color: Colors.white, child: scene),
          ),
        ),
      );
      await tester.pump();
      final boundary =
          key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
      final image = await boundary.toImage();
      final bytes = (await tester.runAsync(
        image.toByteData,
      ))!;
      image.dispose();
      return bytes;
    }

    Widget pair({required double gap, required bool grouped}) {
      Widget shape(double left) => Positioned(
        left: left,
        top: 40,
        child: grouped
            ? const LiquidGlass.grouped(
                shape: LiquidRoundedSuperellipse(borderRadius: 20),
                child: SizedBox(width: 80, height: 50),
              )
            : const LiquidGlass(
                shape: LiquidRoundedSuperellipse(borderRadius: 20),
                child: SizedBox(width: 80, height: 50),
              ),
      );
      final shapes = Stack(children: [shape(20), shape(100 + gap)]);
      return LiquidGlassLayer(
        fake: true,
        defaultAppearance: LiquidGlassAppearance(
          tint: Colors.red.withValues(alpha: .5),
        ),
        child: grouped
            ? LiquidGlassBlendGroup(blend: 24, child: shapes)
            : shapes,
      );
    }

    int redAt(ByteData bytes, int x, int y, int width) =>
        bytes.getUint8((y * width + x) * 4);
    int greenAt(ByteData bytes, int x, int y, int width) =>
        bytes.getUint8((y * width + x) * 4 + 1);

    testWidgets('grouped FakeGlass bridges a small gap', (tester) async {
      tester.view
        ..physicalSize = const Size(240, 140)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);

      // The middle of a 4 px gap, level with the shapes' centers.
      const x = 102;
      const y = 65;
      final bridged = await capture(tester, pair(gap: 4, grouped: true));
      final apart = await capture(tester, pair(gap: 4, grouped: false));
      // Glass tinted red covers the gap only when the shapes blend.
      expect(greenAt(bridged, x, y, 240), lessThan(200));
      expect(redAt(apart, x, y, 240), 255);
      expect(greenAt(apart, x, y, 240), 255);
    });

    testWidgets('grouped FakeGlass far apart draws like ungrouped', (
      tester,
    ) async {
      tester.view
        ..physicalSize = const Size(320, 140)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);

      final grouped = await capture(tester, pair(gap: 90, grouped: true));
      final ungrouped = await capture(tester, pair(gap: 90, grouped: false));
      expect(
        grouped.buffer.asUint8List(),
        orderedEquals(ungrouped.buffer.asUint8List()),
      );
    });
  });
}
