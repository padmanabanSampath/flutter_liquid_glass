import 'package:flutter/material.dart';
import 'package:liquid_glass_renderer_example/pages/slab_sea_glass_page.dart';

/// Slab sea glass after the Sea Glass design concept, recreated with
/// liquid_glass_renderer.
///
/// Run with `flutter run -t lib/slab_sea_glass_main.dart`; a landscape
/// window shows the concept's two-column sheet, a phone a single column.
void main() {
  runApp(
    const MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Sea Glass',
      home: SlabSeaGlassPage(),
    ),
  );
}
