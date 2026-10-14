import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';

/// The glass a [MineralSeaGlassPage] compares mineral sea glass with.
enum MineralPageGlass {
  /// Mineral sea glass: etched, tinted and casting a caustic.
  mineral('Mineral', Icons.grain_rounded),

  /// Tumbled sea glass pebbles.
  pebble('Pebble', Icons.water_drop_rounded),

  /// Milky slab sea glass.
  slab('Slab', Icons.layers_rounded),

  /// Clear liquid glass.
  liquid('Liquid', Icons.blur_on_rounded);

  const MineralPageGlass(this.label, this.icon);

  final String label;
  final IconData icon;
}

/// Mineral sea glass beside the other glasses: a draggable pill over large
/// type, with the etch, the caustic and the mineral tint to play with.
class MineralSeaGlassPage extends StatefulWidget {
  const MineralSeaGlassPage({super.key});

  @override
  State<MineralSeaGlassPage> createState() => MineralSeaGlassPageState();
}

@visibleForTesting
class MineralSeaGlassPageState extends State<MineralSeaGlassPage> {
  MineralPageGlass _glass = MineralPageGlass.mineral;
  double _etch = 0.8;
  double _caustic = 0.7;
  MineralTint _tint = MineralTint.seafoamJade;

  /// Where the pill's center sits, as a share of the stage.
  Offset _pill = const Offset(0.5, 0.5);

  @visibleForTesting
  MineralPageGlass get glass => _glass;

  @visibleForTesting
  MineralTint get tint => _tint;

  /// The settings of the pill's layer for the chosen glass.
  @visibleForTesting
  LiquidGlassSettings get settings => switch (_glass) {
    MineralPageGlass.mineral => LiquidGlassSettings.mineralSeaGlass(
      etch: _etch,
      caustic: _caustic,
    ),
    MineralPageGlass.pebble => LiquidGlassSettings.seaGlass(
      frost: 1 + 6 * _etch,
    ),
    MineralPageGlass.slab => LiquidGlassSettings.slabSeaGlass(),
    MineralPageGlass.liquid => LiquidGlassSettings.ios27Clear(),
  };

  LiquidGlassAppearance get _appearance => switch (_glass) {
    MineralPageGlass.mineral || MineralPageGlass.slab => _tint.appearance,
    MineralPageGlass.pebble => LiquidGlassAppearance.seaGlass(
      color: _tint.swatch,
    ),
    MineralPageGlass.liquid => const LiquidGlassAppearance(),
  };

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFECEDF2),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final size = constraints.biggest;
          final width = math.min<double>(size.width - 32, 420);
          return Stack(
            children: [
              const Positioned.fill(child: _Backdrop()),
              Positioned.fill(
                child: LiquidGlassLayer(
                  settings: settings,
                  child: Stack(
                    children: [
                      Positioned(
                        left: _pill.dx * size.width - width / 2,
                        top: _pill.dy * size.height - 50,
                        width: width,
                        height: 100,
                        child: GestureDetector(
                          onPanUpdate: (details) => setState(() {
                            _pill = Offset(
                              (_pill.dx + details.delta.dx / size.width).clamp(
                                0.1,
                                0.9,
                              ),
                              (_pill.dy + details.delta.dy / size.height).clamp(
                                0.25,
                                0.9,
                              ),
                            );
                          }),
                          child: LiquidGlass(
                            shape: const LiquidRoundedSuperellipse(
                              borderRadius: 50,
                            ),
                            appearance: _appearance,
                            child: _PillContent(
                              ink: _glass == MineralPageGlass.liquid
                                  ? const Color(0xFF1B1D24)
                                  : Colors.white,
                              title: '${_glass.label} glass',
                              subtitle: _glass == MineralPageGlass.liquid
                                  ? 'Clear • Refraction'
                                  : '${_tint.name} • Caustic',
                              icon: _glass.icon,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              SafeArea(
                child: Align(
                  alignment: Alignment.topCenter,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 520),
                      child: _Controls(
                        glass: _glass,
                        etch: _etch,
                        caustic: _caustic,
                        tint: _tint,
                        onGlass: (glass) => setState(() => _glass = glass),
                        onEtch: (value) => setState(() => _etch = value),
                        onCaustic: (value) => setState(() => _caustic = value),
                        onTint: (tint) => setState(() => _tint = tint),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

/// A light, striped wall with large type for the glass to bend.
class _Backdrop extends StatelessWidget {
  const _Backdrop();

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _StripesPainter(),
      child: const Align(
        alignment: Alignment(0, 0.08),
        child: OverflowBox(
          maxWidth: double.infinity,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'ETCHED MINERAL GLASS',
                maxLines: 1,
                style: TextStyle(
                  fontSize: 56,
                  fontWeight: FontWeight.w800,
                  letterSpacing: -1,
                  color: Color(0xFF16181F),
                ),
              ),
              SizedBox(height: 40),
              Text(
                'GLASS  •  CAUSTIC SHADOW  •  GLASS',
                maxLines: 1,
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF1F8A70),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StripesPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = const Color(0x66FFFFFF);
    for (var x = -size.height; x < size.width + size.height; x += 120) {
      canvas.drawPath(
        Path()
          ..moveTo(x, 0)
          ..lineTo(x + 50, 0)
          ..lineTo(x + 50 - size.height * 0.7, size.height)
          ..lineTo(x - size.height * 0.7, size.height)
          ..close(),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _PillContent extends StatelessWidget {
  const _PillContent({
    required this.ink,
    required this.title,
    required this.subtitle,
    required this.icon,
  });

  final Color ink;
  final String title;
  final String subtitle;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const SizedBox(width: 22),
        DecoratedBox(
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.28),
            shape: BoxShape.circle,
          ),
          child: SizedBox.square(
            dimension: 52,
            child: Icon(icon, color: ink, size: 28),
          ),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                  color: ink,
                  shadows: const [
                    Shadow(color: Color(0x40000000), blurRadius: 4),
                  ],
                ),
              ),
              Text(
                subtitle,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 14,
                  color: ink.withValues(alpha: 0.85),
                  shadows: const [
                    Shadow(color: Color(0x33000000), blurRadius: 3),
                  ],
                ),
              ),
            ],
          ),
        ),
        Icon(Icons.open_with, color: ink, size: 24),
        const SizedBox(width: 26),
      ],
    );
  }
}

/// The glass picker, the etch and caustic sliders and the mineral tints.
class _Controls extends StatelessWidget {
  const _Controls({
    required this.glass,
    required this.etch,
    required this.caustic,
    required this.tint,
    required this.onGlass,
    required this.onEtch,
    required this.onCaustic,
    required this.onTint,
  });

  final MineralPageGlass glass;
  final double etch;
  final double caustic;
  final MineralTint tint;
  final ValueChanged<MineralPageGlass> onGlass;
  final ValueChanged<double> onEtch;
  final ValueChanged<double> onCaustic;
  final ValueChanged<MineralTint> onTint;

  @override
  Widget build(BuildContext context) {
    const label = TextStyle(color: Colors.white, fontSize: 14, height: 1.1);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        _Panel(
          child: Row(
            children: [
              for (final option in MineralPageGlass.values)
                Expanded(
                  child: GestureDetector(
                    onTap: () => onGlass(option),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      padding: const EdgeInsets.symmetric(vertical: 10),
                      decoration: BoxDecoration(
                        color: option == glass
                            ? tint.swatch
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(24),
                      ),
                      child: Column(
                        children: [
                          Icon(
                            option.icon,
                            size: 20,
                            color: option == glass
                                ? const Color(0xFF14201C)
                                : Colors.white,
                          ),
                          const SizedBox(height: 2),
                          FittedBox(
                            fit: BoxFit.scaleDown,
                            child: Text(
                              option.label,
                              style: label.copyWith(
                                color: option == glass
                                    ? const Color(0xFF14201C)
                                    : Colors.white,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        _Panel(
          child: Row(
            children: [
              const SizedBox(width: 12),
              const Text('Frost\nEtch', style: label),
              Expanded(
                child: Slider(
                  value: etch,
                  activeColor: tint.swatch,
                  onChanged:
                      glass == MineralPageGlass.mineral ||
                          glass == MineralPageGlass.pebble
                      ? onEtch
                      : null,
                ),
              ),
              const Text('Caustic\nGlow', style: label),
              Expanded(
                child: Slider(
                  value: caustic,
                  activeColor: const Color(0xFFF2C94C),
                  onChanged: glass == MineralPageGlass.mineral
                      ? onCaustic
                      : null,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        _Panel(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
            child: Row(
              children: [
                const Text('Mineral Tint:', style: label),
                const SizedBox(width: 8),
                for (final option in MineralTint.values)
                  Expanded(
                    child: GestureDetector(
                      onTap: () => onTint(option),
                      child: Semantics(
                        label: option.name,
                        selected: option == tint,
                        button: true,
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          margin: const EdgeInsets.all(4),
                          height: 30,
                          decoration: BoxDecoration(
                            color: option.swatch,
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: option == tint
                                  ? Colors.white
                                  : Colors.transparent,
                              width: 3,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _Panel extends StatelessWidget {
  const _Panel({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xE6505257),
        borderRadius: BorderRadius.circular(30),
        boxShadow: const [
          BoxShadow(
            color: Color(0x22000000),
            blurRadius: 16,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: Padding(padding: const EdgeInsets.all(6), child: child),
    );
  }
}
