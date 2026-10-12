import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart' show Icons;
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/common.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// Now playing: frosted controls lying across the lower edge of the album
/// art, so the art refracts through the player card, and transport
/// pebbles joined into one strand of glass.
class MusicPage extends StatefulWidget {
  const MusicPage({super.key});

  @override
  State<MusicPage> createState() => _MusicPageState();
}

class _MusicPageState extends State<MusicPage>
    with AutomaticKeepAliveClientMixin {
  static const _length = 238;
  var _position = 102;
  var _playing = true;
  var _volume = .6;
  var _shuffle = false;
  var _repeat = true;
  Timer? _clock;

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    _clock = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!_playing || !mounted) return;
      setState(() => _position = (_position + 1) % _length);
    });
  }

  @override
  void dispose() {
    _clock?.cancel();
    super.dispose();
  }

  static String _time(int seconds) =>
      '${seconds ~/ 60}:${(seconds % 60).toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final style = ShowcaseScope.of(context);
    return ShowcasePageFrame(
      title: 'Now Playing',
      subtitle: 'Sea glass over album art',
      maxWidth: 440,
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            final art = math.min<double>(constraints.maxWidth, 380);
            return Stack(
              children: [
                // The art lies beneath the glass layer, which refracts it.
                Positioned(
                  top: 0,
                  left: (constraints.maxWidth - art) / 2,
                  width: art,
                  height: art,
                  child: const _AlbumArt(),
                ),
                ShowcaseLayer(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      SizedBox(height: art - 92),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        child: GlassCard(
                          padding: const EdgeInsets.fromLTRB(20, 18, 20, 16),
                          child: _NowPlaying(
                            position: _position,
                            length: _length,
                            time: _time,
                            onSeek: (value) => setState(
                              () => _position = (value * _length).round(),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 26),
                      LiquidGlassBlendGroup(
                        blend: 16,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            GlassPebble(
                              icon: Icons.shuffle_rounded,
                              label: 'Shuffle',
                              size: 46,
                              grouped: true,
                              body: _shuffle ? style.body : null,
                              onTap: () => setState(() => _shuffle = !_shuffle),
                            ),
                            const SizedBox(width: 12),
                            GlassPebble(
                              icon: Icons.skip_previous_rounded,
                              label: 'Previous',
                              size: 60,
                              grouped: true,
                              onTap: () => setState(() => _position = 0),
                            ),
                            const SizedBox(width: 12),
                            GlassPebble(
                              icon: _playing
                                  ? Icons.pause_rounded
                                  : Icons.play_arrow_rounded,
                              label: _playing ? 'Pause' : 'Play',
                              size: 88,
                              grouped: true,
                              body: style.body,
                              onTap: () => setState(() => _playing = !_playing),
                            ),
                            const SizedBox(width: 12),
                            GlassPebble(
                              icon: Icons.skip_next_rounded,
                              label: 'Next',
                              size: 60,
                              grouped: true,
                              onTap: () => setState(() => _position = 0),
                            ),
                            const SizedBox(width: 12),
                            GlassPebble(
                              icon: Icons.repeat_rounded,
                              label: 'Repeat',
                              size: 46,
                              grouped: true,
                              body: _repeat ? style.body : null,
                              onTap: () => setState(() => _repeat = !_repeat),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 26),
                      GlassSlider(
                        value: _volume,
                        label: 'Volume',
                        trailing: Icons.volume_up_rounded,
                        onChanged: (value) => setState(() => _volume = value),
                      ),
                      const SizedBox(height: 28),
                      const SectionHeader('Up next', trailing: '3 songs'),
                      const GlassCard(
                        padding: EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        child: _UpNext(),
                      ),
                    ],
                  ),
                ),
              ],
            );
          },
        ),
      ],
    );
  }
}

class _NowPlaying extends StatelessWidget {
  const _NowPlaying({
    required this.position,
    required this.length,
    required this.time,
    required this.onSeek,
  });

  final int position;
  final int length;
  final String Function(int seconds) time;
  final ValueChanged<double> onSeek;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final ink = style.ink;
    final progress = position / length;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Tumbled Blue',
                    style: TextStyle(
                      color: ink,
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    'The Tide Pool · Low Tide Sessions',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: ink.withValues(alpha: .7),
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.favorite_rounded,
              color: style.body.withValues(alpha: 1),
            ),
          ],
        ),
        const SizedBox(height: 16),
        LayoutBuilder(
          builder: (context, constraints) {
            final width = constraints.maxWidth;
            void seek(Offset local) => onSeek((local.dx / width).clamp(0, 1));
            return GestureDetector(
              behavior: HitTestBehavior.opaque,
              onHorizontalDragDown: (details) => seek(details.localPosition),
              onHorizontalDragUpdate: (details) => seek(details.localPosition),
              child: SizedBox(
                height: 18,
                child: Stack(
                  alignment: Alignment.centerLeft,
                  children: [
                    Container(
                      height: 5,
                      decoration: BoxDecoration(
                        color: ink.withValues(alpha: .16),
                        borderRadius: const BorderRadius.all(
                          Radius.circular(3),
                        ),
                      ),
                    ),
                    Container(
                      width: width * progress,
                      height: 5,
                      decoration: BoxDecoration(
                        color: style.body.withValues(alpha: 1),
                        borderRadius: const BorderRadius.all(
                          Radius.circular(3),
                        ),
                      ),
                    ),
                    Positioned(
                      left: (width * progress - 7).clamp(0, width - 14),
                      child: Container(
                        width: 14,
                        height: 14,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: const Color(0xFFFFFFFF),
                          border: Border.all(
                            color: style.body.withValues(alpha: 1),
                            width: 3,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        ),
        const SizedBox(height: 6),
        Row(
          children: [
            Text(time(position), style: _small(ink)),
            const Spacer(),
            Text('-${time(length - position)}', style: _small(ink)),
          ],
        ),
      ],
    );
  }

  static TextStyle _small(Color ink) => TextStyle(
    color: ink.withValues(alpha: .7),
    fontSize: 12,
    fontWeight: FontWeight.w600,
    fontFeatures: const [FontFeature.tabularFigures()],
  );
}

class _UpNext extends StatelessWidget {
  const _UpNext();

  static const _songs = [
    (
      'Seafoam Static',
      'Marine Layer',
      '3:12',
      [
        Color(0xFF8ED3B6),
        Color(0xFF1F7482),
      ],
    ),
    (
      'Cobalt Hours',
      'The Tide Pool',
      '4:05',
      [
        Color(0xFF2E5AD2),
        Color(0xFFA4C3EC),
      ],
    ),
    (
      'Amber Bottle',
      'Driftwood',
      '2:48',
      [
        Color(0xFFF2AE43),
        Color(0xFF94502A),
      ],
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final ink = ShowcaseScope.of(context).ink;
    return Column(
      children: [
        for (final (title, artist, length, colors) in _songs)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    borderRadius: const BorderRadius.all(Radius.circular(12)),
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: colors,
                    ),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: TextStyle(
                          color: ink,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Text(
                        artist,
                        style: TextStyle(
                          color: ink.withValues(alpha: .65),
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),
                ),
                Text(
                  length,
                  style: TextStyle(
                    color: ink.withValues(alpha: .65),
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

/// Original album art: a low sun over layered swells.
class _AlbumArt extends StatelessWidget {
  const _AlbumArt();

  @override
  Widget build(BuildContext context) {
    return const RepaintBoundary(
      child: ClipRRect(
        borderRadius: BorderRadius.all(Radius.circular(32)),
        child: CustomPaint(painter: _SunsetPainter(), size: Size.infinite),
      ),
    );
  }
}

class _SunsetPainter extends CustomPainter {
  const _SunsetPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final w = size.width;
    final h = size.height;
    canvas.drawRect(
      rect,
      Paint()
        ..shader = const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF3B1E6E), Color(0xFFE8536E), Color(0xFFFFB86B)],
          stops: [0, .5, .72],
        ).createShader(rect),
    );
    final sun = Offset(w * .5, h * .58);
    canvas
      ..drawCircle(
        sun,
        w * .42,
        Paint()
          ..shader = RadialGradient(
            colors: [
              const Color(0xFFFFE29A).withValues(alpha: .8),
              const Color(0x00FFE29A),
            ],
          ).createShader(Rect.fromCircle(center: sun, radius: w * .42)),
      )
      ..drawCircle(sun, w * .17, Paint()..color = const Color(0xFFFFE6A8));
    // Deep water under the sun, shallowing to a sandy beach at the bottom,
    // where the player card lies: light enough for its ink.
    const swells = [
      (.6, Color(0xFF1F7482), .035),
      (.68, Color(0xFF2FB8DA), .04),
      (.77, Color(0xFF8ED3B6), .035),
      (.86, Color(0xFFF3DDB8), .03),
    ];
    for (final (level, color, amplitude) in swells) {
      // Integer steps reach both edges exactly; summing w / 48 stops short
      // and leaves a sliver of sky at the right edge.
      final path = Path()..moveTo(0, h);
      for (var step = 0; step <= 48; step++) {
        final x = w * step / 48;
        path.lineTo(
          x,
          h * level + math.sin(x / w * math.pi * 3 + level * 9) * h * amplitude,
        );
      }
      path
        ..lineTo(w, h)
        ..close();
      canvas.drawPath(path, Paint()..color = color);
    }
  }

  @override
  bool shouldRepaint(_SunsetPainter oldDelegate) => false;
}
