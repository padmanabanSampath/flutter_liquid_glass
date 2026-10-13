import 'dart:math' as math;

import 'package:flutter/material.dart' show Icons;
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/living_glass.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/home_page.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/music_page.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/shade_page.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/tide_pool_page.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/wallet_page.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/weather_page.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pebble_tab_bar.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/sea_glass_menus.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/wallpapers.dart';

/// The sea glass showcase: six screens made of sea glass over soft
/// wallpapers that let the glass carry the scene.
///
/// One style runs through every screen. The palette button in the top
/// trailing corner offers every sea glass color, the style button the
/// wallpaper, finish, tone, shape and shine, and the pebble tab bar
/// switches screens. The tide pool's sun sets where the light comes from.
class SeaGlassShowcasePage extends StatefulWidget {
  const SeaGlassShowcasePage({super.key});

  @override
  State<SeaGlassShowcasePage> createState() => _SeaGlassShowcasePageState();
}

class _SeaGlassShowcasePageState extends State<SeaGlassShowcasePage> {
  final _style = ValueNotifier(const ShowcaseStyle());
  final _pages = PageController();
  var _page = 0;

  /// Set while the pager animates to a tapped tab, so the tab bar does not
  /// step through the pages in between.
  var _jumping = false;

  static const _tabs = <PebbleTab>[
    (icon: Icons.tune_rounded, label: 'Shade'),
    (icon: Icons.waves_rounded, label: 'Tide'),
    (icon: Icons.music_note_rounded, label: 'Music'),
    (icon: Icons.wb_sunny_rounded, label: 'Weather'),
    (icon: Icons.home_rounded, label: 'Home'),
    (icon: Icons.account_balance_wallet_rounded, label: 'Wallet'),
  ];

  @override
  void dispose() {
    _style.dispose();
    _pages.dispose();
    super.dispose();
  }

  Future<void> _goTo(int page) async {
    setState(() {
      _page = page;
      _jumping = true;
    });
    await _pages.animateToPage(
      page,
      duration: const Duration(milliseconds: 480),
      curve: Curves.easeOutCubic,
    );
    _jumping = false;
  }

  void _restyle(ShowcaseStyle Function(ShowcaseStyle style) change) =>
      _style.value = change(_style.value);

  @override
  Widget build(BuildContext context) {
    return ShowcaseScope(
      style: _style,
      child: ValueListenableBuilder(
        valueListenable: _style,
        builder: (context, style, _) {
          final safe = MediaQuery.paddingOf(context);
          return ShowcaseClock(
            running: style.shape == ShowcaseShape.alive,
            child: AnnotatedRegion<SystemUiOverlayStyle>(
              value: style.dark
                  ? SystemUiOverlayStyle.light
                  : SystemUiOverlayStyle.dark,
              child: ColoredBox(
                color: const Color(0xFF000000),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 700),
                      layoutBuilder: (current, previous) => Stack(
                        fit: StackFit.expand,
                        children: [...previous, ?current],
                      ),
                      child: WallpaperView(
                        key: ValueKey(style.wallpaper),
                        wallpaper: style.wallpaper,
                        accent: style.accent,
                      ),
                    ),
                    IgnorePointer(
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 120),
                        color: Color.fromRGBO(
                          0,
                          0,
                          0,
                          math.pow(1 - style.brightness, 2) * .55,
                        ),
                      ),
                    ),
                    PageView(
                      controller: _pages,
                      onPageChanged: (page) {
                        if (!_jumping) setState(() => _page = page);
                      },
                      children: const [
                        ShadePage(),
                        TidePoolPage(),
                        MusicPage(),
                        WeatherPage(),
                        HomePage(),
                        WalletPage(),
                      ],
                    ),
                    Positioned(
                      left: 0,
                      right: 0,
                      bottom: safe.bottom + 18,
                      child: Center(
                        child: PebbleTabBar(
                          tabs: _tabs,
                          selected: _page,
                          onSelected: _goTo,
                        ),
                      ),
                    ),
                    Positioned.fill(
                      child: LiquidGlassLayer(
                        settings: style.settings,
                        defaultAppearance: style.frost,
                        child: SeaGlassControls(
                          top: true,
                          accentIndex: style.accentIndex,
                          tone: style.tone,
                          finish: style.finish,
                          wallpaper: style.wallpaper,
                          shape: style.shape,
                          shine: style.shine,
                          clarity: style.clarity,
                          material: style.material,
                          shadows: style.shadows,
                          frosting: style.frosting,
                          colorStrength: style.colorStrength,
                          onAccent: (index) => _restyle(
                            (style) => style.copyWith(accentIndex: index),
                          ),
                          onTone: (tone) =>
                              _restyle((style) => style.copyWith(tone: tone)),
                          onFinish: (finish) => _restyle(
                            (style) => style.copyWith(finish: finish),
                          ),
                          onWallpaper: (wallpaper) => _restyle(
                            (style) => style.copyWith(wallpaper: wallpaper),
                          ),
                          onShape: (shape) =>
                              _restyle((style) => style.copyWith(shape: shape)),
                          onShine: (shine) =>
                              _restyle((style) => style.copyWith(shine: shine)),
                          onClarity: (clarity) => _restyle(
                            (style) => style.copyWith(clarity: clarity),
                          ),
                          onMaterial: (material) => _restyle(
                            (style) => style.copyWith(
                              material: material,
                              // Slab glass reads best on the concept's pale
                              // paper; a wallpaper chosen by hand stays.
                              wallpaper: switch (style.wallpaper) {
                                ShowcaseWallpaper.seaMist
                                    when material.isSlab =>
                                  ShowcaseWallpaper.paper,
                                ShowcaseWallpaper.paper when !material.isSlab =>
                                  ShowcaseWallpaper.seaMist,
                                final wallpaper => wallpaper,
                              },
                            ),
                          ),
                          onFrosting: (frosting) => _restyle(
                            (style) => style.copyWith(frosting: frosting),
                          ),
                          onShadows: (shadows) => _restyle(
                            (style) => style.copyWith(shadows: shadows),
                          ),
                          onColorStrength: (strength) => _restyle(
                            (style) => style.copyWith(colorStrength: strength),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
