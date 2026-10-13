import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';

/// The time living glass ([LiquidBlob.alive]) flows by, in seconds.
///
/// Ticks only while [running] and the platform allows motion, so round and
/// tumbled pebbles cost nothing. Widgets that read [timeOf] rebuild every
/// frame while it ticks.
class ShowcaseClock extends StatefulWidget {
  const ShowcaseClock({required this.running, required this.child, super.key});

  final bool running;
  final Widget child;

  /// The current time; rebuilds [context] while the clock ticks.
  static double timeOf(BuildContext context) =>
      context
          .dependOnInheritedWidgetOfExactType<_ClockScope>()
          ?.notifier
          ?.value ??
      0;

  @override
  State<ShowcaseClock> createState() => _ShowcaseClockState();
}

class _ShowcaseClockState extends State<ShowcaseClock>
    with SingleTickerProviderStateMixin {
  final _time = ValueNotifier<double>(0);

  /// The time the ticker last started at; a ticker counts from zero on
  /// every start, and living glass must not jump back.
  var _start = 0.0;

  late final Ticker _ticker = createTicker(
    (elapsed) => _time.value = _start + elapsed.inMicroseconds / 1e6,
  );

  void _sync() {
    final run = widget.running && !MediaQuery.disableAnimationsOf(context);
    if (run && !_ticker.isActive) {
      _start = _time.value;
      _ticker.start();
    } else if (!run && _ticker.isActive) {
      _ticker.stop();
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _sync();
  }

  @override
  void didUpdateWidget(ShowcaseClock oldWidget) {
    super.didUpdateWidget(oldWidget);
    _sync();
  }

  @override
  void dispose() {
    _ticker.dispose();
    _time.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) =>
      _ClockScope(time: _time, child: widget.child);
}

class _ClockScope extends InheritedNotifier<ValueNotifier<double>> {
  const _ClockScope({
    required ValueNotifier<double> time,
    required super.child,
  }) : super(notifier: time);
}

/// The outline of the round control [seed] in [shape]. Only living glass
/// follows the [ShowcaseClock].
LiquidShape pebbleOutline(
  BuildContext context,
  ShowcaseShape shape,
  int seed,
) => shape.outline(
  seed,
  shape == ShowcaseShape.alive ? ShowcaseClock.timeOf(context) : 0,
);
