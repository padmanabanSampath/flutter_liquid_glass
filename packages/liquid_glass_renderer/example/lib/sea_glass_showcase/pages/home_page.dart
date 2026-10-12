import 'dart:math' as math;

import 'package:flutter/material.dart' show Icons;
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/common.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// A smart home: rooms strung on glass, a thermostat dial with its buttons
/// fused to the rim, scenes and devices that glow in the body color.
class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage>
    with AutomaticKeepAliveClientMixin {
  static const _rooms = ['Living', 'Kitchen', 'Bedroom', 'Studio'];
  static const _scenes = [
    (Icons.wb_twilight_rounded, 'Morning'),
    (Icons.movie_rounded, 'Movie'),
    (Icons.bedtime_rounded, 'Sleep'),
  ];
  static const _devices = [
    (Icons.light_rounded, 'Ceiling lights', '80% · Warm'),
    (Icons.speaker_rounded, 'Speaker', 'Tumbled Blue'),
    (Icons.blinds_rounded, 'Blinds', 'Half open'),
    (Icons.air_rounded, 'Purifier', 'Quiet mode'),
  ];

  var _room = 0;
  var _scene = 1;
  var _target = 22.5;
  final _on = <int>{0, 1};

  @override
  bool get wantKeepAlive => true;

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final style = ShowcaseScope.of(context);
    return ShowcasePageFrame(
      title: 'Home',
      subtitle: '${_on.length} devices on · ${_target.toStringAsFixed(1)}° set',
      children: [
        ShowcaseLayer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                clipBehavior: Clip.none,
                child: LiquidGlassBlendGroup(
                  blend: 12,
                  child: Row(
                    children: [
                      for (var i = 0; i < _rooms.length; i++) ...[
                        if (i > 0) const SizedBox(width: 8),
                        _Chip(
                          label: _rooms[i],
                          body: i == _room ? style.body : null,
                          onTap: () => setState(() => _room = i),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 28),
              Center(
                child: _Thermostat(
                  target: _target,
                  onChanged: (value) => setState(() => _target = value),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 28),
        ShowcaseLayer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SectionHeader('Scenes'),
              Row(
                children: [
                  for (var i = 0; i < _scenes.length; i++) ...[
                    if (i > 0) const SizedBox(width: 10),
                    Expanded(
                      child: _Scene(
                        icon: _scenes[i].$1,
                        label: _scenes[i].$2,
                        body: i == _scene ? style.body : null,
                        onTap: () => setState(() => _scene = i),
                      ),
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 26),
              SectionHeader(_rooms[_room], trailing: '${_on.length} on'),
              for (var row = 0; row < 2; row++) ...[
                if (row > 0) const SizedBox(height: 12),
                Row(
                  children: [
                    for (var column = 0; column < 2; column++) ...[
                      if (column > 0) const SizedBox(width: 12),
                      Expanded(
                        child: _Device(
                          icon: _devices[row * 2 + column].$1,
                          name: _devices[row * 2 + column].$2,
                          status: _devices[row * 2 + column].$3,
                          body: _on.contains(row * 2 + column)
                              ? style.body
                              : null,
                          onTap: () => setState(() {
                            final index = row * 2 + column;
                            if (!_on.remove(index)) _on.add(index);
                          }),
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip({required this.label, required this.onTap, this.body});

  final String label;
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
      label: label,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: LiquidStretch.seaGlass(
          child: GlassCard(
            grouped: true,
            radius: 22,
            body: body,
            elevation: body == null ? 3 : 5,
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
            child: Text(
              label,
              style: TextStyle(
                color: ink,
                fontSize: 15,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// A frosted dial with the target as an arc of the body color; drag around
/// it or tap the buttons fused to its rim.
class _Thermostat extends StatelessWidget {
  const _Thermostat({required this.target, required this.onChanged});

  final double target;
  final ValueChanged<double> onChanged;

  static const _min = 16.0;
  static const _max = 30.0;
  static const _dial = 232.0;
  static const _button = 58.0;
  // The buttons touch the dial's rim, so the blend fuses them to it.
  static const _width = _dial + 2 * 22;
  static const _height = _dial + 18;

  /// The arc runs clockwise from the lower left (135°) over 270°.
  static const _start = math.pi * .75;
  static const _sweep = math.pi * 1.5;

  void _dragTo(Offset local) {
    const center = Offset(_width / 2, _dial / 2);
    final delta = local - center;
    var angle = math.atan2(delta.dy, delta.dx) - _start;
    while (angle < 0) {
      angle += 2 * math.pi;
    }
    if (angle > _sweep) {
      // In the gap at the bottom: snap to the nearer end.
      angle = angle - _sweep < (2 * math.pi - _sweep) / 2 ? _sweep : 0;
    }
    final value = _min + (_max - _min) * angle / _sweep;
    onChanged((value * 2).round() / 2);
  }

  void _step(double by) => onChanged((target + by).clamp(_min, _max));

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final ink = style.ink;
    final progress = (target - _min) / (_max - _min);
    return SizedBox(
      width: _width,
      height: _height,
      child: LiquidGlassBlendGroup(
        blend: 22,
        child: Stack(
          children: [
            Positioned(
              left: (_width - _dial) / 2,
              top: 0,
              width: _dial,
              height: _dial,
              child: Semantics(
                slider: true,
                label: 'Temperature',
                value: '${target.toStringAsFixed(1)} degrees',
                child: GestureDetector(
                  onPanDown: (details) => _dragTo(
                    details.localPosition +
                        const Offset((_width - _dial) / 2, 0),
                  ),
                  onPanUpdate: (details) => _dragTo(
                    details.localPosition +
                        const Offset((_width - _dial) / 2, 0),
                  ),
                  child: LiquidGlass.grouped(
                    shape: const LiquidOval(),
                    shadows: SeaGlassFinish.shadows(frostedWhite, elevation: 8),
                    child: CustomPaint(
                      painter: _DialPainter(
                        progress: progress,
                        color: style.body.withValues(alpha: 1),
                        track: ink.withValues(alpha: .14),
                        ticks: ink.withValues(alpha: .3),
                      ),
                      child: Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              'HEATING',
                              style: TextStyle(
                                color: ink.withValues(alpha: .65),
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 1.6,
                              ),
                            ),
                            Text(
                              '${target.toStringAsFixed(1)}°',
                              style: TextStyle(
                                color: ink,
                                fontSize: 52,
                                height: 1.1,
                                fontWeight: FontWeight.w300,
                                letterSpacing: -1.5,
                              ),
                            ),
                            Text(
                              'Inside 21.8°',
                              style: TextStyle(
                                color: ink.withValues(alpha: .7),
                                fontSize: 13,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
            Positioned(
              left: 0,
              top: _height - _button - 8,
              child: GlassPebble(
                icon: Icons.remove_rounded,
                label: 'Cooler',
                size: _button,
                grouped: true,
                onTap: () => _step(-.5),
              ),
            ),
            Positioned(
              right: 0,
              top: _height - _button - 8,
              child: GlassPebble(
                icon: Icons.add_rounded,
                label: 'Warmer',
                size: _button,
                grouped: true,
                body: style.body,
                onTap: () => _step(.5),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DialPainter extends CustomPainter {
  const _DialPainter({
    required this.progress,
    required this.color,
    required this.track,
    required this.ticks,
  });

  final double progress;
  final Color color;
  final Color track;
  final Color ticks;

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.shortestSide / 2 - 22;
    final rect = Rect.fromCircle(center: center, radius: radius);
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 10;
    canvas
      ..drawArc(
        rect,
        _Thermostat._start,
        _Thermostat._sweep,
        false,
        stroke..color = track,
      )
      ..drawArc(
        rect,
        _Thermostat._start,
        _Thermostat._sweep * progress,
        false,
        stroke..color = color,
      );
    final tick = Paint()
      ..color = ticks
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round;
    for (var i = 0; i <= 28; i++) {
      final angle = _Thermostat._start + _Thermostat._sweep * i / 28;
      final direction = Offset(math.cos(angle), math.sin(angle));
      final inner = radius - (i % 4 == 0 ? 22 : 17);
      canvas.drawLine(
        center + direction * inner,
        center + direction * (radius - 12),
        tick,
      );
    }
    final knobAngle = _Thermostat._start + _Thermostat._sweep * progress;
    final knob =
        center + Offset(math.cos(knobAngle), math.sin(knobAngle)) * radius;
    canvas
      ..drawCircle(knob, 11, Paint()..color = const Color(0xFFFFFFFF))
      ..drawCircle(knob, 6, Paint()..color = color);
  }

  @override
  bool shouldRepaint(_DialPainter oldDelegate) =>
      oldDelegate.progress != progress ||
      oldDelegate.color != color ||
      oldDelegate.track != track ||
      oldDelegate.ticks != ticks;
}

class _Scene extends StatelessWidget {
  const _Scene({
    required this.icon,
    required this.label,
    required this.onTap,
    this.body,
  });

  final IconData icon;
  final String label;
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
      label: '$label scene',
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: LiquidStretch.seaGlass(
          child: GlassCard(
            radius: 24,
            body: body,
            elevation: body == null ? 3 : 6,
            padding: const EdgeInsets.symmetric(vertical: 14),
            child: Column(
              children: [
                Icon(icon, color: ink, size: 24),
                const SizedBox(height: 6),
                Text(
                  label,
                  style: TextStyle(
                    color: ink,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _Device extends StatelessWidget {
  const _Device({
    required this.icon,
    required this.name,
    required this.status,
    required this.onTap,
    this.body,
  });

  final IconData icon;
  final String name;
  final String status;
  final Color? body;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final body = this.body;
    final on = body != null;
    final ink = on ? inkOn(body) : style.ink;
    return Semantics(
      button: true,
      toggled: on,
      label: name,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: LiquidStretch.seaGlass(
          child: GlassCard(
            body: body,
            elevation: on ? 6 : 3,
            padding: const EdgeInsets.all(16),
            child: SizedBox(
              height: 112,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      IconBubble(
                        icon: icon,
                        size: 42,
                        iconColor: ink,
                        color: on ? ink.withValues(alpha: .16) : null,
                      ),
                      const Spacer(),
                      _Switch(on: on, ink: ink),
                    ],
                  ),
                  const Spacer(),
                  Text(
                    name,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: ink,
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    on ? status : 'Off',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: ink.withValues(alpha: .7),
                      fontSize: 13,
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

class _Switch extends StatelessWidget {
  const _Switch({required this.on, required this.ink});

  final bool on;
  final Color ink;

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 220),
      width: 40,
      height: 24,
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(
        color: ink.withValues(alpha: on ? .3 : .14),
        borderRadius: const BorderRadius.all(Radius.circular(12)),
      ),
      child: AnimatedAlign(
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOutBack,
        alignment: on ? Alignment.centerRight : Alignment.centerLeft,
        child: Container(
          width: 18,
          height: 18,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: on ? ink : const Color(0xFFFFFFFF),
          ),
        ),
      ),
    );
  }
}
