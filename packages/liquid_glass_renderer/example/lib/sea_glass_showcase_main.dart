import 'package:flutter/cupertino.dart';
import 'package:liquid_glass_renderer_example/performance_probe.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_page.dart';

const _enablePerformanceProbe = bool.fromEnvironment(
  'LIQUID_GLASS_EXAMPLE_PERFORMANCE_PROBE',
);

/// The sea glass showcase.
///
/// Run with `flutter run -t lib/sea_glass_showcase_main.dart`. Add
/// `--dart-define=LIQUID_GLASS_EXAMPLE_PERFORMANCE_PROBE=true` to log frame
/// timings every two seconds.
void main() {
  runApp(
    const CupertinoApp(
      debugShowCheckedModeBanner: false,
      title: 'Sea Glass',
      home: _enablePerformanceProbe
          ? Stack(children: [SeaGlassShowcasePage(), PerformanceProbe()])
          : SeaGlassShowcasePage(),
    ),
  );
}
