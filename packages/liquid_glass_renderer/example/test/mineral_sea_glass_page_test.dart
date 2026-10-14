import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/pages/mineral_sea_glass_page.dart';

Future<MineralSeaGlassPageState> _pumpPage(
  WidgetTester tester,
  Size size,
) async {
  tester.view
    ..physicalSize = size * 2
    ..devicePixelRatio = 2;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(
    const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MineralSeaGlassPage(),
    ),
  );
  await tester.pump(const Duration(milliseconds: 300));
  return tester.state<MineralSeaGlassPageState>(
    find.byType(MineralSeaGlassPage),
  );
}

void main() {
  for (final (name, size) in [
    ('phone', const Size(402, 874)),
    ('wide', const Size(1280, 800)),
  ]) {
    testWidgets('switches glass and tint on a $name screen', (tester) async {
      final page = await _pumpPage(tester, size);
      expect(page.glass, MineralPageGlass.mineral);
      expect(page.settings.seaGlassFinish.effectiveCaustic, greaterThan(0));
      expect(page.settings.seaGlassFinish.effectiveGlitter, greaterThan(0));

      await tester.tap(find.bySemanticsLabel('Cerulean Aqua'));
      await tester.pump(const Duration(milliseconds: 300));
      expect(page.tint, MineralTint.ceruleanAqua);

      for (final glass in MineralPageGlass.values.reversed) {
        await tester.tap(find.text(glass.label));
        await tester.pump(const Duration(milliseconds: 300));
        expect(page.glass, glass);
      }
      expect(page.settings, isNot(LiquidGlassSettings.slabSeaGlass()));
    });
  }

  testWidgets('the pill follows a drag', (tester) async {
    await _pumpPage(tester, const Size(402, 874));
    final pill = find.byType(LiquidGlass);
    final before = tester.getCenter(pill);
    await tester.drag(pill, const Offset(0, -120));
    await tester.pump();
    expect(tester.getCenter(pill).dy, lessThan(before.dy - 80));
  });

  testWidgets('only mineral glass lights a caustic', (tester) async {
    final page = await _pumpPage(tester, const Size(402, 874));
    final sliders = find.byType(Slider);
    expect(tester.widget<Slider>(sliders.last).onChanged, isNotNull);
    await tester.tap(find.text(MineralPageGlass.slab.label));
    await tester.pump(const Duration(milliseconds: 300));
    expect(page.glass, MineralPageGlass.slab);
    expect(tester.widget<Slider>(sliders.first).onChanged, isNull);
    expect(tester.widget<Slider>(sliders.last).onChanged, isNull);
  });
}
