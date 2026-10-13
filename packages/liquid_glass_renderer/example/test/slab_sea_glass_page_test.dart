import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/pages/slab_sea_glass_page.dart';

const _wide = Size(1920, 1080);

Future<SlabSeaGlassPageState> _pumpPage(
  WidgetTester tester, {
  Size size = _wide,
  double devicePixelRatio = 1,
}) async {
  tester.view
    ..physicalSize = size * devicePixelRatio
    ..devicePixelRatio = devicePixelRatio;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(
    const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: SlabSeaGlassPage(),
    ),
  );
  // The glass settles from its intro.
  await _frames(tester, 40);
  return tester.state<SlabSeaGlassPageState>(
    find.byType(SlabSeaGlassPage),
  );
}

Future<void> _frames(WidgetTester tester, int count) async {
  for (var i = 0; i < count; i++) {
    await tester.pump(const Duration(milliseconds: 50));
  }
}

/// Unmounts the page and lets its timers run out.
Future<void> _close(WidgetTester tester) async {
  await tester.pumpWidget(const SizedBox());
  await tester.pump(const Duration(seconds: 4));
}

void main() {
  testWidgets('daylight, dusk and the flashlight light the glass', (
    tester,
  ) async {
    final page = await _pumpPage(tester);
    final day = page.settingsFor(_wide);
    expect(day.seaGlassFinish.shading, 1);
    expect(day.seaGlassFinish.elevation, closeTo(22, 0.01));
    expect(day.light.position, isNull);
    expect(day.light.ambient, const Color(0xFF000000));

    await tester.tap(find.byIcon(Icons.dark_mode_outlined));
    await _frames(tester, 4);
    final dusk = page.settingsFor(_wide);
    expect(dusk.light.position, isNull);
    expect(dusk.light.color.b, lessThan(dusk.light.color.r));
    expect(dusk.light.ambient, isNot(const Color(0xFF000000)));

    await tester.tap(find.byIcon(Icons.flashlight_off));
    await _frames(tester, 4);
    final flashlight = page.settingsFor(_wide);
    expect(flashlight.light.position, isNotNull);
    expect(flashlight.light.reach, isNotNull);
    await _close(tester);
  });

  testWidgets('the sparkle cycles the glass sets and Undo restores one', (
    tester,
  ) async {
    final page = await _pumpPage(tester);
    expect(page.glass, SlabSeaGlass.original);
    await tester.tap(find.byIcon(Icons.auto_awesome_rounded));
    await _frames(tester, 10);
    expect(page.glass, SlabSeaGlass.beach);
    expect(find.text(SlabSeaGlass.beach.name), findsOneWidget);
    await tester.tap(find.text('Undo'));
    await _frames(tester, 10);
    expect(page.glass, SlabSeaGlass.original);
    await _close(tester);
  });

  for (final (name, size, devicePixelRatio) in [
    ('wide', _wide, 1.0),
    ('phone', const Size(402, 874), 2.0),
  ]) {
    testWidgets('the menu, search and reply open on a $name screen', (
      tester,
    ) async {
      await _pumpPage(
        tester,
        size: size,
        devicePixelRatio: devicePixelRatio,
      );
      await tester.tap(find.byIcon(Icons.play_arrow));
      await _frames(tester, 6);
      expect(find.byIcon(Icons.pause), findsOneWidget);
      await tester.tap(find.byIcon(Icons.pause));

      await tester.tap(find.byIcon(Icons.add));
      await _frames(tester, 20);
      await tester.tap(find.text('Drawing'));
      await _frames(tester, 10);
      expect(find.text('New drawing · demo'), findsOneWidget);

      await tester.tap(find.text('Search'));
      await _frames(tester, 16);
      const question = 'Why is sea glass frosted?';
      await tester.tap(find.text(question));
      await _frames(tester, 60);
      expect(find.textContaining('Decades of waves'), findsOneWidget);
      await _close(tester);
    });
  }
}
