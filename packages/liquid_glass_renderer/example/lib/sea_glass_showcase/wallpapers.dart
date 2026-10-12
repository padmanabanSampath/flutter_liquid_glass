import 'dart:math' as math;

import 'package:flutter/widgets.dart';

/// A soft glow of color in a wallpaper.
@immutable
class WallpaperGlow {
  const WallpaperGlow(
    this.color,
    this.center,
    this.radius, {
    this.drift = const Offset(.05, .04),
    this.phase = 0,
  });

  final Color color;

  /// Where the glow sits, as a fraction of the wallpaper's size.
  final Offset center;

  /// How far the glow reaches, as a fraction of the wallpaper's longest side.
  final double radius;

  /// How far the glow wanders, as a fraction of the wallpaper's size.
  final Offset drift;

  /// Where in the drift cycle the glow starts, from 0 to 1.
  final double phase;
}

/// The backgrounds of the sea glass showcase.
///
/// Soft color fields elevate the glass instead of competing with it: the
/// frost picks up their glows, and each one flatters a different family of
/// sea glass colors. Sea Mist follows the selected glass color. Bloom is an
/// original wallpaper in the style of iOS's bold ribbon wallpapers. The
/// photos come from the playground, for comparison with a busy backdrop; add
/// your own image to `assets/backdrops/` and list it here to use it.
enum ShowcaseWallpaper {
  seaMist('Sea Mist'),
  lagoon(
    'Lagoon',
    top: Color(0xFFE3F6F1),
    bottom: Color(0xFFA8DCD6),
    glows: [
      WallpaperGlow(Color(0x992FB8DA), Offset(.15, .25), .45),
      WallpaperGlow(Color(0x998ED3B6), Offset(.85, .55), .5, phase: .3),
      WallpaperGlow(Color(0x80A4C3EC), Offset(.3, .85), .4, phase: .6),
      WallpaperGlow(Color(0x66FFC9A8), Offset(.75, .15), .3, phase: .8),
    ],
  ),
  coralSunset(
    'Coral Sunset',
    top: Color(0xFFFFE3CF),
    bottom: Color(0xFFF4A7A3),
    glows: [
      WallpaperGlow(Color(0xB3F2AE43), Offset(.2, .2), .45),
      WallpaperGlow(Color(0x99FF6F91), Offset(.85, .4), .45, phase: .25),
      WallpaperGlow(Color(0x80B9A2E8), Offset(.3, .75), .5, phase: .5),
      WallpaperGlow(Color(0x80FFD27A), Offset(.8, .9), .35, phase: .75),
    ],
  ),
  lavenderDusk(
    'Lavender Dusk',
    top: Color(0xFFF1EAFB),
    bottom: Color(0xFFC3CCF3),
    glows: [
      WallpaperGlow(Color(0xB3C9A7F0), Offset(.2, .3), .45),
      WallpaperGlow(Color(0x99A4C3EC), Offset(.8, .5), .5, phase: .3),
      WallpaperGlow(Color(0x80F7B6D2), Offset(.35, .85), .4, phase: .55),
      WallpaperGlow(Color(0x668ED3B6), Offset(.85, .15), .3, phase: .8),
    ],
  ),
  meadow(
    'Meadow',
    top: Color(0xFFF4F7DA),
    bottom: Color(0xFFBFE2B8),
    glows: [
      WallpaperGlow(Color(0xB3C4D37C), Offset(.2, .25), .45),
      WallpaperGlow(Color(0x803E9B5E), Offset(.85, .6), .45, phase: .35),
      WallpaperGlow(Color(0x80F2AE43), Offset(.7, .15), .3, phase: .6),
      WallpaperGlow(Color(0x998ED3B6), Offset(.25, .85), .45, phase: .85),
    ],
  ),
  aurora(
    'Aurora',
    dark: true,
    top: Color(0xFF050B1A),
    bottom: Color(0xFF0A2533),
    glows: [
      WallpaperGlow(Color(0xCC2EE6A6), Offset(.25, .3), .5),
      WallpaperGlow(Color(0x9926B5C9), Offset(.75, .45), .45, phase: .3),
      WallpaperGlow(Color(0xB37B5CFF), Offset(.6, .15), .4, phase: .55),
      WallpaperGlow(Color(0x80E255C9), Offset(.2, .8), .4, phase: .8),
    ],
  ),
  deepSea(
    'Deep Sea',
    dark: true,
    top: Color(0xFF03111F),
    bottom: Color(0xFF07375A),
    glows: [
      WallpaperGlow(Color(0xCC2E5AD2), Offset(.2, .35), .5),
      WallpaperGlow(Color(0xB32FB8DA), Offset(.8, .6), .45, phase: .3),
      WallpaperGlow(Color(0x808ED3B6), Offset(.35, .9), .4, phase: .6),
      WallpaperGlow(Color(0x991F7482), Offset(.85, .15), .35, phase: .85),
    ],
  ),
  bloom(
    'Bloom',
    dark: true,
    top: Color(0xFF0B1240),
    bottom: Color(0xFF2A0E45),
  ),
  coast('Coast', photo: 'assets/backdrops/coast.webp'),
  nightCity('Night City', dark: true, photo: 'assets/backdrops/city.webp');

  const ShowcaseWallpaper(
    this.label, {
    this.dark = false,
    this.top = const Color(0xFFFFFFFF),
    this.bottom = const Color(0xFFFFFFFF),
    this.glows = const [],
    this.photo,
  });

  final String label;

  /// Whether content on this wallpaper needs light ink.
  final bool dark;

  final Color top;
  final Color bottom;
  final List<WallpaperGlow> glows;

  /// The asset of a photo wallpaper.
  final String? photo;

  /// A bright pastel of [color] at [lightness], keeping its hue: mixing a
  /// deep color with white turns it grey, while this stays airy.
  static Color _pastel(Color color, double lightness) {
    final hsl = HSLColor.fromColor(color.withValues(alpha: 1));
    return hsl
        .withLightness(lightness)
        .withSaturation(math.min(1, hsl.saturation * 1.15 + .1))
        .toColor();
  }

  /// The glows of Sea Mist, which follow [accent].
  static List<WallpaperGlow> _mistGlows(Color accent) => [
    WallpaperGlow(
      _pastel(accent, .62).withValues(alpha: .55),
      const Offset(.2, .3),
      .5,
    ),
    const WallpaperGlow(Color(0x59FFC3A0), Offset(.8, .45), .35, phase: .3),
    WallpaperGlow(
      _pastel(accent, .7).withValues(alpha: .5),
      const Offset(.7, .85),
      .45,
      phase: .55,
    ),
    const WallpaperGlow(Color(0x40D8C7DB), Offset(.15, .75), .35, phase: .8),
  ];
}

/// Fills its space with [wallpaper], its glows slowly drifting.
///
/// [accent] tints Sea Mist and animates to each new color.
class WallpaperView extends StatefulWidget {
  const WallpaperView({
    required this.wallpaper,
    required this.accent,
    this.animate = true,
    super.key,
  });

  final ShowcaseWallpaper wallpaper;
  final Color accent;
  final bool animate;

  @override
  State<WallpaperView> createState() => _WallpaperViewState();
}

class _WallpaperViewState extends State<WallpaperView>
    with SingleTickerProviderStateMixin {
  late final AnimationController _drift = AnimationController(
    vsync: this,
    duration: const Duration(seconds: 28),
  );

  @override
  void initState() {
    super.initState();
    _syncDrift();
  }

  @override
  void didUpdateWidget(WallpaperView oldWidget) {
    super.didUpdateWidget(oldWidget);
    _syncDrift();
  }

  void _syncDrift() {
    final drifts =
        widget.animate && widget.wallpaper.glows.isNotEmpty ||
        widget.animate && widget.wallpaper == ShowcaseWallpaper.seaMist;
    if (drifts && !_drift.isAnimating) {
      _drift.repeat();
    } else if (!drifts) {
      _drift.stop();
    }
  }

  @override
  void dispose() {
    _drift.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final wallpaper = widget.wallpaper;
    if (wallpaper.photo case final photo?) {
      return Image.asset(
        photo,
        fit: BoxFit.cover,
        width: double.infinity,
        height: double.infinity,
      );
    }
    if (wallpaper == ShowcaseWallpaper.bloom) {
      // Static: its soft edges are blurred, which is too costly per frame.
      return const RepaintBoundary(
        child: CustomPaint(painter: _BloomPainter(), size: Size.infinite),
      );
    }
    return TweenAnimationBuilder<Color?>(
      tween: ColorTween(end: widget.accent),
      duration: const Duration(milliseconds: 600),
      curve: Curves.easeOutCubic,
      builder: (context, accent, _) => CustomPaint(
        painter: _GlowPainter(
          wallpaper: wallpaper,
          accent: accent ?? widget.accent,
          drift: _drift,
        ),
        size: Size.infinite,
      ),
    );
  }
}

/// A small round preview of [wallpaper] for pickers.
class WallpaperThumbnail extends StatelessWidget {
  const WallpaperThumbnail({
    required this.wallpaper,
    required this.accent,
    super.key,
  });

  final ShowcaseWallpaper wallpaper;
  final Color accent;

  @override
  Widget build(BuildContext context) {
    final Widget preview;
    if (wallpaper.photo case final photo?) {
      preview = Image.asset(photo, fit: BoxFit.cover, cacheWidth: 160);
    } else if (wallpaper == ShowcaseWallpaper.bloom) {
      preview = const CustomPaint(painter: _BloomPainter());
    } else {
      preview = CustomPaint(
        painter: _GlowPainter(wallpaper: wallpaper, accent: accent),
      );
    }
    return ClipOval(child: SizedBox.expand(child: preview));
  }
}

class _GlowPainter extends CustomPainter {
  _GlowPainter({required this.wallpaper, required this.accent, this.drift})
    : super(repaint: drift);

  final ShowcaseWallpaper wallpaper;
  final Color accent;
  final Animation<double>? drift;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final mist = wallpaper == ShowcaseWallpaper.seaMist;
    final top = mist ? ShowcaseWallpaper._pastel(accent, .91) : wallpaper.top;
    final bottom = mist
        ? ShowcaseWallpaper._pastel(accent, .8)
        : wallpaper.bottom;
    canvas.drawRect(
      rect,
      Paint()
        ..shader = LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [top, bottom],
        ).createShader(rect),
    );
    final t = drift?.value ?? 0;
    final reach = size.longestSide;
    final glows = mist ? ShowcaseWallpaper._mistGlows(accent) : wallpaper.glows;
    for (final glow in glows) {
      final angle = 2 * math.pi * (t + glow.phase);
      final center = Offset(
        (glow.center.dx + math.sin(angle) * glow.drift.dx) * size.width,
        (glow.center.dy + math.cos(angle) * glow.drift.dy) * size.height,
      );
      final radius = glow.radius * reach;
      canvas.drawCircle(
        center,
        radius,
        Paint()
          ..shader = RadialGradient(
            colors: [glow.color, glow.color.withValues(alpha: 0)],
          ).createShader(Rect.fromCircle(center: center, radius: radius)),
      );
    }
  }

  @override
  bool shouldRepaint(_GlowPainter oldDelegate) =>
      oldDelegate.wallpaper != wallpaper || oldDelegate.accent != accent;
}

/// Bold, softly lit ribbons of color sweeping across a deep indigo field.
class _BloomPainter extends CustomPainter {
  const _BloomPainter();

  static const _ribbons = [
    // (start height, end height, bulge, thickness, colors)
    (.18, .42, -.16, .2, [Color(0xFF6C3BFF), Color(0xFF2F6BFF)]),
    (.42, .30, .22, .2, [Color(0xFFFF3D8B), Color(0xFFE040FB)]),
    (.62, .70, -.20, .22, [Color(0xFFFF8A3D), Color(0xFFFF3D8B)]),
    (.86, .78, .18, .2, [Color(0xFF22D3EE), Color(0xFF2F6BFF)]),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    canvas.drawRect(
      rect,
      Paint()
        ..shader = const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0B1240), Color(0xFF2A0E45)],
        ).createShader(rect),
    );
    final w = size.width;
    final h = size.height;
    final blur = size.shortestSide * .035;
    for (final (start, end, bulge, thickness, colors) in _ribbons) {
      final band = h * thickness;
      final path = Path()
        ..moveTo(-w * .2, h * start)
        ..cubicTo(
          w * .3,
          h * (start + bulge),
          w * .7,
          h * (end - bulge),
          w * 1.2,
          h * end,
        )
        ..lineTo(w * 1.2, h * end + band)
        ..cubicTo(
          w * .7,
          h * (end - bulge) + band,
          w * .3,
          h * (start + bulge) + band,
          -w * .2,
          h * start + band,
        )
        ..close();
      final bounds = path.getBounds();
      canvas
        ..drawPath(
          path,
          Paint()
            ..shader = LinearGradient(colors: colors).createShader(bounds)
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, blur),
        )
        // A lit leading edge, as if each ribbon were a sheet of glass.
        ..drawPath(
          path,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = size.shortestSide * .006
            ..color = const Color(0x66FFFFFF)
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, blur * .25),
        );
    }
  }

  @override
  bool shouldRepaint(_BloomPainter oldDelegate) => false;
}
