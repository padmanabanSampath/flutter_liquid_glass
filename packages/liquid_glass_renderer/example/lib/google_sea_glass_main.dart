import 'package:flutter/material.dart';
import 'package:liquid_glass_renderer_example/pages/google_sea_glass_page.dart';

/// Google's Sea Glass concept, recreated with liquid_glass_renderer.
///
/// Run with `flutter run -t lib/google_sea_glass_main.dart`; a landscape
/// window shows the concept's two-column sheet, a phone a single column.
void main() {
  runApp(
    const MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Sea Glass',
      home: GoogleSeaGlassPage(),
    ),
  );
}
