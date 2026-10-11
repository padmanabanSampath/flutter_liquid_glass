import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/playground/scenes/draggable_glass.dart';

/// Shapes with different appearances in one layer and one blend group.
///
/// Appearance is per shape, so light, dark, clear and tinted glass share a
/// single backdrop capture, and their colors blend where the shapes merge.
/// Under a sea glass finish the swatches are sea glass pebbles instead, which
/// fuse through a thinner, paler neck.
class ColorsScene extends StatelessWidget {
  const ColorsScene({required this.blend, super.key});

  final ValueListenable<double> blend;

  static const _swatches = [
    (
      label: 'Light',
      offset: Offset(-50, -50),
      glyph: Color(0xFF000000),
      appearance: LiquidGlassAppearance.ios27RegularLight(),
    ),
    (
      label: 'Dark',
      offset: Offset(50, -50),
      glyph: Color(0xFFFFFFFF),
      appearance: LiquidGlassAppearance.ios27RegularDark(),
    ),
    (
      label: 'Clear',
      offset: Offset(-50, 50),
      glyph: Color(0xFFFFFFFF),
      appearance: LiquidGlassAppearance.ios27Clear(),
    ),
    (
      label: 'Blue',
      offset: Offset(50, 50),
      glyph: Color(0xFFFFFFFF),
      appearance: LiquidGlassAppearance.ios27RegularLight(
        tint: Color(0xFF0A84FF),
      ),
    ),
  ];

  static const _pebbles = [
    (
      label: 'Seafoam',
      offset: Offset(-50, -50),
      glyph: Color(0xFF173A40),
      appearance: LiquidGlassAppearance.seaGlass(),
    ),
    (
      label: 'Cobalt',
      offset: Offset(50, -50),
      glyph: Color(0xFFFFFFFF),
      appearance: LiquidGlassAppearance.seaGlass(color: SeaGlassColors.cobalt),
    ),
    (
      label: 'Amber',
      offset: Offset(-50, 50),
      glyph: Color(0xFF3A2405),
      appearance: LiquidGlassAppearance.seaGlass(color: SeaGlassColors.amber),
    ),
    (
      label: 'Aqua',
      offset: Offset(50, 50),
      glyph: Color(0xFFFFFFFF),
      appearance: LiquidGlassAppearance.seaGlass(color: SeaGlassColors.aqua),
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final swatches = LiquidGlassSettings.of(context).seaGlassFinish.isEnabled
        ? _pebbles
        : _swatches;
    return LayoutBuilder(
      builder: (context, constraints) {
        final center = constraints.biggest.center(Offset.zero);
        return ValueListenableBuilder(
          valueListenable: blend,
          builder: (context, blend, shapes) =>
              LiquidGlassBlendGroup(blend: blend, child: shapes!),
          child: Stack(
            children: [
              for (final swatch in swatches)
                DraggableGlass(
                  center: center,
                  offset: swatch.offset,
                  size: const Size.square(108),
                  shape: const LiquidOval(),
                  grouped: true,
                  appearance: swatch.appearance,
                  child: Text(
                    swatch.label,
                    style: TextStyle(
                      color: swatch.glyph,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
}
