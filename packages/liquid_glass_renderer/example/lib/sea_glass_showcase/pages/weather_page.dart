import 'dart:math' as math;

import 'package:flutter/material.dart' show Icons;
import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/common.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// The weather: a cloud of sea glass drifting across the sun, hourly
/// pebbles and a week of forecasts.
class WeatherPage extends StatefulWidget {
  const WeatherPage({super.key});

  @override
  State<WeatherPage> createState() => _WeatherPageState();
}

class _WeatherPageState extends State<WeatherPage>
    with AutomaticKeepAliveClientMixin {
  var _hour = 0;

  @override
  bool get wantKeepAlive => true;

  static const _hours = [
    ('Now', Icons.wb_sunny_rounded, 27),
    ('1 PM', Icons.wb_sunny_rounded, 29),
    ('2 PM', Icons.wb_cloudy_rounded, 30),
    ('3 PM', Icons.wb_cloudy_rounded, 30),
    ('4 PM', Icons.water_drop_rounded, 28),
    ('5 PM', Icons.thunderstorm_rounded, 26),
  ];

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final style = ShowcaseScope.of(context);
    final ink = style.ink;
    return ShowcasePageFrame(
      title: 'Weather',
      subtitle: 'Kochi · Sea breeze from the west',
      children: [
        SizedBox(
          height: 190,
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      '${_hours[_hour].$3}°',
                      style: TextStyle(
                        color: ink,
                        fontSize: 96,
                        height: 1,
                        fontWeight: FontWeight.w200,
                        letterSpacing: -4,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _hour < 2 ? 'Mostly sunny' : 'Clouds rolling in',
                      style: TextStyle(
                        color: ink,
                        fontSize: 19,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'H 31°   L 24°',
                      style: TextStyle(
                        color: ink.withValues(alpha: .7),
                        fontSize: 15,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 190, child: _SunAndCloud()),
            ],
          ),
        ),
        const SizedBox(height: 24),
        ShowcaseLayer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SectionHeader('Hourly', trailing: 'Tap an hour'),
              Row(
                children: [
                  for (var i = 0; i < _hours.length; i++) ...[
                    if (i > 0) const SizedBox(width: 8),
                    Expanded(
                      child: _HourPebble(
                        label: _hours[i].$1,
                        icon: _hours[i].$2,
                        temperature: _hours[i].$3,
                        body: i == _hour ? style.body : null,
                        onTap: () => setState(() => _hour = i),
                      ),
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 26),
              const SectionHeader('Conditions'),
              const Row(
                children: [
                  Expanded(
                    child: _DetailCard(
                      title: 'Wind',
                      value: '14 km/h',
                      note: 'From the west',
                      icon: Icons.air_rounded,
                      gauge: _Gauge.compass,
                    ),
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: _DetailCard(
                      title: 'Humidity',
                      value: '68%',
                      note: 'Dew point 21°',
                      icon: Icons.water_drop_outlined,
                      gauge: _Gauge.bar,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const Row(
                children: [
                  Expanded(
                    child: _DetailCard(
                      title: 'UV index',
                      value: '6',
                      note: 'High until 3 PM',
                      icon: Icons.wb_sunny_outlined,
                      gauge: _Gauge.ring,
                    ),
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: _DetailCard(
                      title: 'Sunset',
                      value: '6:12 PM',
                      note: 'Golden hour 5:40',
                      icon: Icons.wb_twilight_rounded,
                      gauge: _Gauge.arc,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 26),
              const SectionHeader('7-day forecast'),
              const GlassCard(
                padding: EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                child: _Week(),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// A sun painted on the wallpaper and a cloud of four puffs of living sea
/// glass that drift, billow and merge in front of it.
class _SunAndCloud extends StatefulWidget {
  const _SunAndCloud();

  @override
  State<_SunAndCloud> createState() => _SunAndCloudState();
}

class _SunAndCloudState extends State<_SunAndCloud>
    with SingleTickerProviderStateMixin {
  /// Seconds since the cloud appeared. The puffs drift once around every
  /// nine seconds, and their outlines billow as living glass.
  final _time = ValueNotifier<double>(0);
  late final Ticker _ticker = createTicker(
    (elapsed) => _time.value = elapsed.inMicroseconds / 1e6,
  )..start();

  static const _driftPeriod = 9.0;

  /// Center, size and drift phase of each puff, in a 190 x 190 box.
  static const _puffs = [
    (Offset(70, 118), Size(92, 62), 0.0),
    (Offset(112, 96), Size(96, 84), .25),
    (Offset(150, 122), Size(72, 54), .5),
    (Offset(96, 132), Size(110, 48), .75),
  ];

  @override
  void dispose() {
    _ticker.dispose();
    _time.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        const Positioned(
          left: 58,
          top: 46,
          width: 110,
          height: 110,
          child: CustomPaint(painter: _SunPainter()),
        ),
        Positioned.fill(
          child: ShowcaseLayer(
            child: ValueListenableBuilder(
              valueListenable: _time,
              builder: (context, time, _) => LiquidGlassBlendGroup(
                blend: 26,
                child: Stack(
                  children: [
                    for (final (i, (center, size, phase)) in _puffs.indexed)
                      _puff(center, size, phase, time, i),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _puff(Offset center, Size size, double phase, double time, int i) {
    final angle = 2 * math.pi * (time / _driftPeriod + phase);
    final c = center + Offset(math.sin(angle) * 7, math.cos(angle) * 4);
    final breathe = 1 + .06 * math.sin(angle * 2);
    final w = size.width * breathe;
    final h = size.height * breathe;
    return Positioned(
      left: c.dx - w / 2,
      top: c.dy - h / 2,
      width: w,
      height: h,
      child: LiquidGlass.grouped(
        shape: LiquidBlob.alive(time: time * .8, seed: 11 + i),
        appearance: const LiquidGlassAppearance.seaGlass(
          color: Color(0x99F4F8F6),
        ),
        shadows: SeaGlassFinish.shadows(frostedWhite, elevation: 4),
        child: const SizedBox.expand(),
      ),
    );
  }
}

class _SunPainter extends CustomPainter {
  const _SunPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.shortestSide / 2;
    canvas
      ..drawCircle(
        center,
        radius,
        Paint()
          ..shader = RadialGradient(
            colors: [
              const Color(0xFFFFC94D).withValues(alpha: .65),
              const Color(0x00FFC94D),
            ],
          ).createShader(Rect.fromCircle(center: center, radius: radius)),
      )
      ..drawCircle(
        center,
        radius * .48,
        Paint()
          ..shader = const RadialGradient(
            colors: [Color(0xFFFFE9A6), Color(0xFFFFB238)],
          ).createShader(Rect.fromCircle(center: center, radius: radius * .48)),
      );
  }

  @override
  bool shouldRepaint(_SunPainter oldDelegate) => false;
}

class _HourPebble extends StatelessWidget {
  const _HourPebble({
    required this.label,
    required this.icon,
    required this.temperature,
    required this.onTap,
    this.body,
  });

  final String label;
  final IconData icon;
  final int temperature;
  final Color? body;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final body = this.body;
    final ink = body == null ? style.ink : inkOn(body);
    return Semantics(
      button: true,
      selected: body != null,
      label: '$label, $temperature degrees',
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: SizedBox(
          height: 124,
          child: LiquidStretch.seaGlass(
            child: GlassCard(
              radius: 30,
              padding: const EdgeInsets.symmetric(vertical: 14),
              body: body,
              elevation: body == null ? 3 : 6,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  FittedBox(
                    child: Text(
                      label,
                      style: TextStyle(
                        color: ink.withValues(alpha: .8),
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  Icon(icon, color: ink, size: 24),
                  Text(
                    '$temperature°',
                    style: TextStyle(
                      color: ink,
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

enum _Gauge { compass, bar, ring, arc }

class _DetailCard extends StatelessWidget {
  const _DetailCard({
    required this.title,
    required this.value,
    required this.note,
    required this.icon,
    required this.gauge,
  });

  final String title;
  final String value;
  final String note;
  final IconData icon;
  final _Gauge gauge;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final ink = style.ink;
    return GlassCard(
      radius: 26,
      padding: const EdgeInsets.all(16),
      child: SizedBox(
        height: 124,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: ink.withValues(alpha: .75), size: 16),
                const SizedBox(width: 6),
                Text(
                  title.toUpperCase(),
                  style: TextStyle(
                    color: ink.withValues(alpha: .75),
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Expanded(
                  child: FittedBox(
                    alignment: Alignment.centerLeft,
                    fit: BoxFit.scaleDown,
                    child: Text(
                      value,
                      style: TextStyle(
                        color: ink,
                        fontSize: 26,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
                SizedBox.square(
                  dimension: 46,
                  child: CustomPaint(
                    painter: _GaugePainter(
                      gauge: gauge,
                      color: style.body.withValues(alpha: 1),
                      track: ink.withValues(alpha: .16),
                    ),
                  ),
                ),
              ],
            ),
            const Spacer(),
            Text(
              note,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: ink.withValues(alpha: .7), fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  const _GaugePainter({
    required this.gauge,
    required this.color,
    required this.track,
  });

  final _Gauge gauge;
  final Color color;
  final Color track;

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.shortestSide / 2 - 4;
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.round;
    switch (gauge) {
      case _Gauge.compass:
        canvas.drawCircle(center, radius, stroke..color = track);
        final tip = center + const Offset(-1, .2) * radius * .8;
        canvas
          ..drawLine(
            center - (tip - center),
            tip,
            stroke
              ..color = color
              ..strokeWidth = 4,
          )
          ..drawCircle(tip, 4, Paint()..color = color);
      case _Gauge.bar:
        for (var i = 0; i < 5; i++) {
          final x = size.width * (i + .5) / 5;
          final filled = i < 3.4;
          canvas.drawLine(
            Offset(x, size.height - 4),
            Offset(x, size.height - 4 - (size.height - 8) * (i + 1) / 5),
            stroke
              ..strokeWidth = 5
              ..color = filled ? color : track,
          );
        }
      case _Gauge.ring:
        final rect = Rect.fromCircle(center: center, radius: radius);
        canvas
          ..drawArc(
            rect,
            math.pi * .75,
            math.pi * 1.5,
            false,
            stroke..color = track,
          )
          ..drawArc(
            rect,
            math.pi * .75,
            math.pi * 1.5 * 6 / 11,
            false,
            stroke..color = color,
          );
      case _Gauge.arc:
        final rect = Rect.fromCircle(
          center: center + Offset(0, radius * .5),
          radius: radius,
        );
        canvas.drawArc(rect, math.pi, math.pi, false, stroke..color = track);
        const done = .78;
        canvas.drawArc(
          rect,
          math.pi,
          math.pi * done,
          false,
          stroke..color = color,
        );
        const angle = math.pi * (1 + done);
        canvas.drawCircle(
          rect.center + Offset(math.cos(angle), math.sin(angle)) * radius,
          5,
          Paint()..color = const Color(0xFFFFB238),
        );
    }
  }

  @override
  bool shouldRepaint(_GaugePainter oldDelegate) =>
      oldDelegate.gauge != gauge ||
      oldDelegate.color != color ||
      oldDelegate.track != track;
}

class _Week extends StatelessWidget {
  const _Week();

  static const _days = [
    ('Today', Icons.wb_sunny_rounded, 24, 31),
    ('Tue', Icons.wb_cloudy_rounded, 24, 30),
    ('Wed', Icons.thunderstorm_rounded, 23, 28),
    ('Thu', Icons.water_drop_rounded, 22, 27),
    ('Fri', Icons.wb_cloudy_rounded, 23, 29),
    ('Sat', Icons.wb_sunny_rounded, 24, 32),
    ('Sun', Icons.wb_sunny_rounded, 25, 33),
  ];

  static const _min = 22;
  static const _max = 33;

  @override
  Widget build(BuildContext context) {
    final ink = ShowcaseScope.of(context).ink;
    final number = TextStyle(
      color: ink,
      fontSize: 15,
      fontWeight: FontWeight.w600,
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    return Column(
      children: [
        for (final (day, icon, low, high) in _days)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 9),
            child: Row(
              children: [
                SizedBox(
                  width: 58,
                  child: Text(
                    day,
                    style: TextStyle(
                      color: ink,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Icon(icon, color: ink, size: 20),
                const SizedBox(width: 14),
                SizedBox(
                  width: 30,
                  child: Text(
                    '$low°',
                    style: number.copyWith(color: ink.withValues(alpha: .6)),
                  ),
                ),
                Expanded(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final width = constraints.maxWidth;
                      final start = (low - _min) / (_max - _min) * width;
                      final end = (high - _min) / (_max - _min) * width;
                      return SizedBox(
                        height: 6,
                        child: Stack(
                          children: [
                            Positioned.fill(
                              child: DecoratedBox(
                                decoration: BoxDecoration(
                                  color: ink.withValues(alpha: .12),
                                  borderRadius: const BorderRadius.all(
                                    Radius.circular(3),
                                  ),
                                ),
                              ),
                            ),
                            Positioned(
                              left: start,
                              width: end - start,
                              top: 0,
                              bottom: 0,
                              child: const DecoratedBox(
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.all(
                                    Radius.circular(3),
                                  ),
                                  gradient: LinearGradient(
                                    colors: [
                                      Color(0xFF2FB8DA),
                                      Color(0xFFF2AE43),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
                SizedBox(
                  width: 40,
                  child: Text(
                    '$high°',
                    textAlign: TextAlign.right,
                    style: number,
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}
