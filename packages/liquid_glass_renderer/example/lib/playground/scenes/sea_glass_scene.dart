import 'package:flutter/cupertino.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/playground/scenes/draggable_glass.dart';
import 'package:liquid_glass_renderer_example/playground/sheet_avoidance.dart';

/// A quick settings panel in sea glass: frosted tiles that turn to deep teal
/// glass when switched on, and a brightness slider.
///
/// Every tile is one shape in the stage's layer. Inactive tiles inherit the
/// layer's appearance; active tiles carry a teal body color, so switching a
/// tile only changes its appearance and never needs another backdrop
/// capture.
class SeaGlassScene extends StatefulWidget {
  const SeaGlassScene({super.key});

  /// The appearance of switched-on tiles: a teal sea glass body, or teal
  /// tinted glass when the layer has no sea glass finish.
  static LiquidGlassAppearance activeAppearance(BuildContext context) =>
      LiquidGlassSettings.of(context).seaGlassFinish.isEnabled
      ? const LiquidGlassAppearance.seaGlass(color: SeaGlassColors.teal)
      : LiquidGlassAppearance.ios27Regular(
          brightness: CupertinoTheme.brightnessOf(context),
          tint: SeaGlassColors.teal,
        );

  @override
  State<SeaGlassScene> createState() => _SeaGlassSceneState();
}

/// Ink on frosted tiles, which follows the appearance, and on teal tiles.
Color _ink(BuildContext context, {bool active = false}) =>
    active || CupertinoTheme.brightnessOf(context) == Brightness.dark
    ? const Color(0xFFFFFFFF)
    : const Color(0xFF173A40);

const _tileHeight = 64.0;
const _gap = 12.0;

class _SeaGlassSceneState extends State<SeaGlassScene> {
  final _active = <String>{'Wi-Fi', 'Location'};
  var _brightness = 0.55;

  void _toggle(String label) => setState(() {
    if (!_active.remove(label)) _active.add(label);
  });

  Widget _wide(IconData icon, String label, [String? detail]) => _Tile(
    icon: icon,
    label: label,
    detail: detail,
    active: _active.contains(label),
    onTap: () => _toggle(label),
  );

  Widget _small(IconData icon, String label) => _Tile(
    icon: icon,
    label: label,
    compact: true,
    active: _active.contains(label),
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
                  onChanged: (value) => setState(() => _brightness = value),
                ),
                const SizedBox(height: _gap * 2),
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
                  _wide(CupertinoIcons.recordingtape, 'Screen record', 'Start'),
                ]),
                _row([
                  _row([
                    _small(CupertinoIcons.airplane, 'Airplane mode'),
                    _small(CupertinoIcons.qrcode_viewfinder, 'Scan QR code'),
                  ], bottom: 0),
                  _wide(CupertinoIcons.share, 'Quick Share'),
                ], bottom: 0),
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
    required this.active,
    required this.onTap,
    this.detail,
    this.compact = false,
  });

  final IconData icon;
  final String label;
  final String? detail;
  final bool active;
  final bool compact;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final ink = _ink(context, active: active);
    return Semantics(
      button: true,
      toggled: active,
      label: compact ? label : null,
      child: GestureDetector(
        onTap: onTap,
        child: LiquidStretch(
          child: LiquidGlass(
            shape: const LiquidRoundedSuperellipse(
              borderRadius: _tileHeight / 2,
            ),
            appearance: active ? SeaGlassScene.activeAppearance(context) : null,
            child: GlowContent(
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

/// A brightness slider: a teal glass bar for the level and a frosted bar
/// for the rest, split by a thin handle.
class _BrightnessSlider extends StatelessWidget {
  const _BrightnessSlider({required this.value, required this.onChanged});

  final double value;
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
                      appearance: SeaGlassScene.activeAppearance(context),
                      child: const SizedBox.expand(),
                    ),
                  ),
                  Positioned(
                    left: level + _handleGap / 2 - 2,
                    top: 6,
                    bottom: 6,
                    width: 4,
                    child: const DecoratedBox(
                      decoration: BoxDecoration(
                        color: SeaGlassColors.teal,
                        borderRadius: BorderRadius.all(Radius.circular(2)),
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
