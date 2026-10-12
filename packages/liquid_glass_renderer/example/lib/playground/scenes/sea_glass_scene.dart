import 'package:flutter/cupertino.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/playground/scenes/draggable_glass.dart';
import 'package:liquid_glass_renderer_example/playground/sheet_avoidance.dart';

/// A quick settings panel in sea glass: frosted tiles that turn to colored
/// glass when switched on, a brightness slider, and a row of accent colors.
///
/// Every tile is one shape in the stage's layer. Inactive tiles inherit the
/// layer's appearance; active tiles carry the accent as their body color, so
/// switching a tile only changes its appearance and never needs another
/// backdrop capture.
///
/// The tiles, and the accent pebbles, are blend groups: at rest they keep
/// apart, but a tile pulled toward its neighbor flows into it like a water
/// drop and pinches off again when let go.
class SeaGlassScene extends StatefulWidget {
  const SeaGlassScene({super.key});

  /// The accent colors to pick from, teal first. With the twelve shapes of
  /// the panel and the stage's settings button, three swatches fill the
  /// stage layer's sixteen shapes, so the whole scene still samples the
  /// backdrop once.
  static const accents = [
    ('Teal', SeaGlassColors.teal),
    ('Aqua', SeaGlassColors.aqua),
    ('Amber', SeaGlassColors.amber),
  ];

  /// Blend radius of the tiles: below their gap, so they keep apart at rest
  /// and bridge once a tile is pulled half the gap toward a neighbor.
  static const tileBlend = 16.0;

  /// Blend radius of the accent pebbles, which sit further apart.
  static const accentBlend = 22.0;

  /// The appearance of switched-on tiles: a sea glass body of [color], or
  /// glass tinted [color] when the layer has no sea glass finish.
  static LiquidGlassAppearance activeAppearance(
    BuildContext context, [
    Color color = SeaGlassColors.teal,
  ]) => LiquidGlassSettings.of(context).seaGlassFinish.isEnabled
      ? LiquidGlassAppearance.seaGlass(color: color)
      : LiquidGlassAppearance.ios27Regular(
          brightness: CupertinoTheme.brightnessOf(context),
          tint: color,
        );

  @override
  State<SeaGlassScene> createState() => _SeaGlassSceneState();
}

/// Ink on frosted tiles, which follows the appearance, and on tiles in
/// [accent]: white on deep colors, dark on light ones.
Color _ink(BuildContext context, {Color? accent}) {
  final light = accent != null
      ? accent.computeLuminance() > 0.35
      : CupertinoTheme.brightnessOf(context) == Brightness.light;
  return light ? const Color(0xFF173A40) : const Color(0xFFFFFFFF);
}

/// Squash and stretch for a tile: Apple's gel-like give for liquid glass,
/// a heavier, wobbling drop for sea glass.
Widget _stretch(BuildContext context, Widget child) =>
    LiquidGlassSettings.of(context).seaGlassFinish.isEnabled
    ? LiquidStretch.seaGlass(child: child)
    : LiquidStretch(child: child);

const _tileHeight = 64.0;
const _gap = 12.0;

class _SeaGlassSceneState extends State<SeaGlassScene> {
  final _active = <String>{'Wi-Fi', 'Location'};
  var _brightness = 0.55;
  var _accent = SeaGlassColors.teal;

  void _toggle(String label) => setState(() {
    if (!_active.remove(label)) _active.add(label);
  });

  Widget _wide(IconData icon, String label, [String? detail]) => _Tile(
    icon: icon,
    label: label,
    detail: detail,
    accent: _active.contains(label) ? _accent : null,
    onTap: () => _toggle(label),
  );

  Widget _small(IconData icon, String label) => _Tile(
    icon: icon,
    label: label,
    compact: true,
    accent: _active.contains(label) ? _accent : null,
    onTap: () => _toggle(label),
  );

  @override
  Widget build(BuildContext context) {
    return AvoidSheet(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 400),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _BrightnessSlider(
                  value: _brightness,
                  accent: _accent,
                  onChanged: (value) => setState(() => _brightness = value),
                ),
                const SizedBox(height: _gap * 2),
                LiquidGlassBlendGroup(
                  blend: SeaGlassScene.tileBlend,
                  child: Column(
                    children: [
                      _row([
                        _wide(CupertinoIcons.wifi, 'Wi-Fi', 'Home 5G'),
                        _wide(
                          CupertinoIcons.antenna_radiowaves_left_right,
                          'Mobile data',
                        ),
                      ]),
                      _row([
                        _wide(CupertinoIcons.bluetooth, 'Bluetooth'),
                        _row([
                          _small(CupertinoIcons.lightbulb, 'Torch'),
                          _small(CupertinoIcons.location_fill, 'Location'),
                        ], bottom: 0),
                      ]),
                      _row([
                        _wide(CupertinoIcons.moon, 'Modes'),
                        _wide(
                          CupertinoIcons.recordingtape,
                          'Screen record',
                          'Start',
                        ),
                      ]),
                      _row([
                        _row([
                          _small(CupertinoIcons.airplane, 'Airplane mode'),
                          _small(
                            CupertinoIcons.qrcode_viewfinder,
                            'Scan QR code',
                          ),
                        ], bottom: 0),
                        _wide(CupertinoIcons.share, 'Quick Share'),
                      ], bottom: 0),
                    ],
                  ),
                ),
                const SizedBox(height: _gap * 2),
                _AccentPicker(
                  selected: _accent,
                  onSelected: (color) => setState(() => _accent = color),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  /// Two equal halves, [_gap] apart.
  Widget _row(List<Widget> halves, {double bottom = _gap}) => Padding(
    padding: EdgeInsets.only(bottom: bottom),
    child: Row(
      children: [
        Expanded(child: halves[0]),
        const SizedBox(width: _gap),
        Expanded(child: halves[1]),
      ],
    ),
  );
}

class _Tile extends StatelessWidget {
  const _Tile({
    required this.icon,
    required this.label,
    required this.onTap,
    this.accent,
    this.detail,
    this.compact = false,
  });

  final IconData icon;
  final String label;
  final String? detail;

  /// The body color while switched on; `null` while off.
  final Color? accent;
  final bool compact;
  final VoidCallback onTap;

  bool get active => accent != null;

  @override
  Widget build(BuildContext context) {
    final ink = _ink(context, accent: accent);
    return Semantics(
      button: true,
      toggled: active,
      label: compact ? label : null,
      child: GestureDetector(
        onTap: onTap,
        child: _stretch(
          context,
          LiquidGlass.grouped(
            shape: const LiquidRoundedSuperellipse(
              borderRadius: _tileHeight / 2,
            ),
            appearance: accent == null
                ? null
                : SeaGlassScene.activeAppearance(context, accent!),
            child: _FrostedGlow(
              child: SizedBox(
                height: _tileHeight,
                child: compact
                    ? Center(child: Icon(icon, color: ink, size: 24))
                    : Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 20),
                        child: Row(
                          children: [
                            Icon(icon, color: ink, size: 24),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    label,
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: TextStyle(
                                      color: ink,
                                      fontSize: 16,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  if (detail case final detail?)
                                    Text(
                                      detail,
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: TextStyle(
                                        color: ink.withValues(alpha: .75),
                                        fontSize: 13,
                                      ),
                                    ),
                                ],
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
    );
  }
}

/// The touch glow of a frosted tile: the frost spreads the light through the
/// body, so it is wider and softer than on clear glass.
class _FrostedGlow extends StatelessWidget {
  const _FrostedGlow({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) => GlassGlowLayer(
    child: GlassGlow(
      glowColor: const Color(0x33FFFFFF),
      glowRadius: 1.6,
      child: child,
    ),
  );
}

/// A brightness slider: a teal glass bar for the level and a frosted bar
/// for the rest, split by a thin handle.
class _BrightnessSlider extends StatelessWidget {
  const _BrightnessSlider({
    required this.value,
    required this.accent,
    required this.onChanged,
  });

  final double value;
  final Color accent;
  final ValueChanged<double> onChanged;

  static const _height = 56.0;
  static const _handleGap = 10.0;
  static const _minBar = _height;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final travel = width - 2 * _minBar - _handleGap;
        final level = _minBar + travel * value;
        void update(Offset local) =>
            onChanged(((local.dx - _minBar) / travel).clamp(0.0, 1.0));
        return Semantics(
          slider: true,
          label: 'Brightness',
          value: '${(value * 100).round()}%',
          child: ImmediateDrag(
            onUpdate: (details) => update(details.localPosition),
            child: SizedBox(
              height: _height,
              child: Stack(
                children: [
                  Positioned(
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: level,
                    child: LiquidGlass(
                      shape: const LiquidRoundedSuperellipse(
                        borderRadius: _height / 2,
                      ),
                      appearance: SeaGlassScene.activeAppearance(
                        context,
                        accent,
                      ),
                      child: const SizedBox.expand(),
                    ),
                  ),
                  Positioned(
                    left: level + _handleGap / 2 - 2,
                    top: 6,
                    bottom: 6,
                    width: 4,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: accent,
                        borderRadius: const BorderRadius.all(
                          Radius.circular(2),
                        ),
                      ),
                    ),
                  ),
                  Positioned(
                    left: level + _handleGap,
                    right: 0,
                    top: 0,
                    bottom: 0,
                    child: LiquidGlass(
                      shape: const LiquidRoundedSuperellipse(
                        borderRadius: _height / 2,
                      ),
                      child: Align(
                        alignment: Alignment.centerRight,
                        child: Padding(
                          padding: const EdgeInsets.only(right: 18),
                          child: Icon(
                            CupertinoIcons.sun_max,
                            color: _ink(context),
                            size: 22,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

/// Small glass pebbles to pick the accent the switched-on tiles are made of.
/// Pulled toward each other, they merge, their colors flowing together.
class _AccentPicker extends StatelessWidget {
  const _AccentPicker({required this.selected, required this.onSelected});

  final Color selected;
  final ValueChanged<Color> onSelected;

  static const _size = 40.0;

  @override
  Widget build(BuildContext context) {
    return LiquidGlassBlendGroup(
      blend: SeaGlassScene.accentBlend,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          for (final (name, color) in SeaGlassScene.accents)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8),
              child: Semantics(
                button: true,
                selected: color == selected,
                label: name,
                child: GestureDetector(
                  onTap: () => onSelected(color),
                  child: _stretch(
                    context,
                    LiquidGlass.grouped(
                      shape: const LiquidOval(),
                      appearance: SeaGlassScene.activeAppearance(
                        context,
                        color,
                      ),
                      child: SizedBox.square(
                        dimension: _size,
                        child: color == selected
                            ? Icon(
                                CupertinoIcons.checkmark_alt,
                                size: 20,
                                color: _ink(context, accent: color),
                              )
                            : null,
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
