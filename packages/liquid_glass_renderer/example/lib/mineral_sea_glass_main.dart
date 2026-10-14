import 'package:flutter/material.dart';
import 'package:liquid_glass_renderer_example/pages/mineral_sea_glass_page.dart';

/// Mineral sea glass beside pebble, slab and liquid glass: drag the pill,
/// etch it, light its caustic and pick a mineral tint.
///
/// Run with `flutter run -t lib/mineral_sea_glass_main.dart`.
void main() {
  runApp(
    const MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Mineral Sea Glass',
      home: MineralSeaGlassPage(),
    ),
  );
}
