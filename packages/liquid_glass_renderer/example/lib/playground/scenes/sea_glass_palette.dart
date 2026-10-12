import 'package:flutter/cupertino.dart';
import 'package:flutter/scheduler.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/playground/scenes/draggable_glass.dart';

/// The sea glass palette as tumbled pebbles, each its own irregular shape,
/// above them a pebble that is alive.
///
/// All pebbles share one blend group: drag one into another and they fuse
/// like drops, the colors meeting at the seam. With the stage's settings
/// button the scene draws fourteen shapes, within one layer's sixteen.
class SeaGlassPalette extends StatelessWidget {
  const SeaGlassPalette({required this.center, super.key});

  /// The middle of the stage, which the pebbles are placed around.
  final Offset center;

  /// The colors in the order the palette shows them.
  static const colors = [
    ('Seafoam', SeaGlassColors.seafoam),
    ('Kelly', SeaGlassColors.kellyGreen),
    ('Citron', SeaGlassColors.citron),
    ('Aqua', SeaGlassColors.aqua),
    ('Cornflower', SeaGlassColors.cornflower),
    ('Cobalt', SeaGlassColors.cobalt),
    ('Lavender', SeaGlassColors.lavender),
    ('Ruby', SeaGlassColors.ruby),
    ('Amber', SeaGlassColors.amber),
    ('Root beer', SeaGlassColors.brown),
    ('Clear', SeaGlassColors.clear),
    ('Mist', SeaGlassColors.mist),
  ];

  static const _columns = 4;
  static const _pebble = Size(72, 62);

  /// Distance between pebble centers: the gaps stay wider than the default
  /// blend radius, so the pebbles keep apart until dragged together.
  static const _cell = Size(88, 84);

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        AlivePebble(center: center, offset: const Offset(0, -137)),
        for (final (index, (label, color)) in colors.indexed)
          DraggableGlass(
            center: center,
            offset: Offset(
              (index % _columns - (_columns - 1) / 2) * _cell.width,
              13 + index ~/ _columns * _cell.height,
            ),
            size: _pebble,
            shape: LiquidBlob.pebble(seed: index),
            grouped: true,
            appearance: LiquidGlassAppearance.seaGlass(color: color),
            child: _Label(label),
          ),
      ],
    );
  }
}

/// A sea glass pebble whose outline flows and breathes, as
/// [LiquidBlob.alive] at the elapsed time. Holds still when the platform
/// asks for reduced motion.
///
/// Must be a direct child of a [Stack], like [DraggableGlass].
class AlivePebble extends StatefulWidget {
  const AlivePebble({
    required this.center,
    this.offset = Offset.zero,
    this.color = SeaGlassColors.amber,
    super.key,
  });

  final Offset center;
  final Offset offset;
  final Color color;

  static const size = Size(160, 150);

  @override
  State<AlivePebble> createState() => _AlivePebbleState();
}

class _AlivePebbleState extends State<AlivePebble>
    with SingleTickerProviderStateMixin {
  final _time = ValueNotifier<double>(0);
  late final Ticker _ticker = createTicker(
    (elapsed) => _time.value = elapsed.inMicroseconds / 1e6,
  );

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

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder(
      valueListenable: _time,
      builder: (context, time, label) => DraggableGlass(
        center: widget.center,
        offset: widget.offset,
        size: AlivePebble.size,
        shape: LiquidBlob.alive(time: time, seed: 3),
        grouped: true,
        appearance: LiquidGlassAppearance.seaGlass(color: widget.color),
        child: label!,
      ),
      child: const _Label('Alive', fontSize: 17),
    );
  }
}

class _Label extends StatelessWidget {
  const _Label(this.text, {this.fontSize = 11});

  final String text;
  final double fontSize;

  @override
  Widget build(BuildContext context) {
    // Sea glass lets the backdrop through, so the label carries its own
    // contrast: white ink in a soft shadow reads over light and dark alike.
    return Text(
      text,
      style: TextStyle(
        color: const Color(0xFFFFFFFF),
        fontSize: fontSize,
        fontWeight: FontWeight.w600,
        shadows: const [Shadow(color: Color(0x99000000), blurRadius: 4)],
      ),
    );
  }
}
