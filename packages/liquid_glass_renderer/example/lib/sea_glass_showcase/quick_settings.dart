import 'package:flutter/material.dart' show Icons;
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// A quick settings panel in sea glass: frosted tiles that turn to colored
/// glass when switched on, and a brightness slider in the body color.
///
/// The tiles are drops of one blend group: their blend stays below the gap
/// between them, so they keep apart at rest, but drag one toward a neighbor
/// and it stretches, flows into it and pinches off when let go.
///
/// Its twelve shapes all live in the enclosing layer, so the panel samples
/// the backdrop once however many tiles are on.
class QuickSettingsPanel extends StatelessWidget {
  const QuickSettingsPanel({
    required this.body,
    required this.active,
    required this.onToggle,
    required this.brightness,
    required this.onBrightness,
    required this.darkWallpaper,
    this.tileBody,
    super.key,
  });

  /// The sea glass body of the brightness level, and of switched-on
  /// tiles without [tileBody].
  final Color body;

  /// The body of each switched-on tile by its label; [body] by default.
  final Color Function(String label)? tileBody;
  final Set<String> active;
  final ValueChanged<String> onToggle;
  final double brightness;
  final ValueChanged<double> onBrightness;
  final bool darkWallpaper;

  static const tileHeight = 64.0;
  static const gap = 12.0;

  /// Blend radius of the tiles, just below their [gap].
  static const tileBlend = 10.0;

  Widget _tile(
    IconData icon,
    String label, {
    String? detail,
    bool compact = false,
  }) => _Tile(
    icon: icon,
    label: label,
    detail: detail,
    compact: compact,
    body: active.contains(label) ? (tileBody?.call(label) ?? body) : null,
    darkWallpaper: darkWallpaper,
    onTap: () => onToggle(label),
  );

  Widget _row(Widget start, Widget end, {double bottom = gap}) => Padding(
    padding: EdgeInsets.only(bottom: bottom),
    child: Row(
      children: [
        Expanded(child: start),
        const SizedBox(width: gap),
        Expanded(child: end),
      ],
    ),
  );

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        _BrightnessSlider(
          value: brightness,
          body: body,
          darkWallpaper: darkWallpaper,
          onChanged: onBrightness,
        ),
        const SizedBox(height: gap * 2),
        LiquidGlassBlendGroup(
          blend: tileBlend,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _row(
                _tile(Icons.wifi_rounded, 'Wi-Fi', detail: 'Tide Pool 5G'),
                _tile(Icons.swap_vert_rounded, 'Mobile data'),
              ),
              _row(
                _tile(Icons.bluetooth_rounded, 'Bluetooth'),
                _row(
                  _tile(Icons.flashlight_on_rounded, 'Torch', compact: true),
                  _tile(Icons.location_on_rounded, 'Location', compact: true),
                  bottom: 0,
                ),
              ),
              _row(
                _tile(Icons.bedtime_rounded, 'Modes'),
                _tile(
                  Icons.radio_button_checked_rounded,
                  'Screen record',
                  detail: 'Start',
                ),
              ),
              _row(
                _row(
                  _tile(Icons.flight_rounded, 'Airplane mode', compact: true),
                  _tile(
                    Icons.qr_code_scanner_rounded,
                    'Scan QR code',
                    compact: true,
                  ),
                  bottom: 0,
                ),
                _tile(Icons.sync_alt_rounded, 'Quick Share'),
                bottom: 0,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _Tile extends StatelessWidget {
  const _Tile({
    required this.icon,
    required this.label,
    required this.darkWallpaper,
    required this.onTap,
    this.body,
    this.detail,
    this.compact = false,
  });

  final IconData icon;
  final String label;
  final String? detail;
  final bool compact;
  final bool darkWallpaper;
  final VoidCallback onTap;

  /// The sea glass body while switched on; `null` while off.
  final Color? body;

  @override
  Widget build(BuildContext context) {
    final body = this.body;
    final on = body != null;
    final ink = on ? inkOn(body) : inkFor(darkWallpaper: darkWallpaper);
    final iconWidget = Icon(icon, color: ink, size: 24);
    return Semantics(
      button: true,
      toggled: on,
      label: label,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: LiquidStretch.seaGlass(
          child: LiquidGlass.grouped(
            shape: const LiquidRoundedSuperellipse(
              borderRadius: QuickSettingsPanel.tileHeight / 2,
            ),
            appearance: on ? LiquidGlassAppearance.seaGlass(color: body) : null,
            shadows: showcaseShadows(
              context,
              on ? body : frostedWhite,
              elevation: on ? 5 : 3,
            ),
            child: _FrostedGlow(
              child: SizedBox(
                height: QuickSettingsPanel.tileHeight,
                child: compact
                    ? Center(child: iconWidget)
                    : Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 14),
                        child: Row(
                          children: [
                            _IconBubble(
                              visible: !on,
                              darkWallpaper: darkWallpaper,
                              child: iconWidget,
                            ),
                            const SizedBox(width: 10),
                            Expanded(child: _Labels(label, detail, ink)),
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

/// The soft disc behind the icon of a switched-off tile; switched-on tiles
/// show the icon on the colored glass itself.
class _IconBubble extends StatelessWidget {
  const _IconBubble({
    required this.visible,
    required this.darkWallpaper,
    required this.child,
  });

  final bool visible;
  final bool darkWallpaper;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: visible
            ? Color(darkWallpaper ? 0x24FFFFFF : 0x4DFFFFFF)
            : const Color(0x00FFFFFF),
      ),
      child: Center(child: child),
    );
  }
}

class _Labels extends StatelessWidget {
  const _Labels(this.label, this.detail, this.ink);

  final String label;
  final String? detail;
  final Color ink;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            color: ink,
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
        if (detail case final detail?)
          Text(
            detail,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(color: ink.withValues(alpha: .75), fontSize: 12),
          ),
      ],
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

/// A brightness slider: a bar of colored glass for the level and a frosted
/// bar for the rest, split by a thin handle.
class _BrightnessSlider extends StatelessWidget {
  const _BrightnessSlider({
    required this.value,
    required this.body,
    required this.darkWallpaper,
    required this.onChanged,
  });

  final double value;
  final Color body;
  final bool darkWallpaper;
  final ValueChanged<double> onChanged;

  static const _height = 56.0;
  static const _handleGap = 10.0;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final travel = width - 2 * _height - _handleGap;
        final level = _height + travel * value;
        void update(Offset local) =>
            onChanged(((local.dx - _height) / travel).clamp(0.0, 1.0));
        return Semantics(
          slider: true,
          label: 'Brightness',
          value: '${(value * 100).round()}%',
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onHorizontalDragDown: (details) => update(details.localPosition),
            onHorizontalDragUpdate: (details) => update(details.localPosition),
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
                      appearance: LiquidGlassAppearance.seaGlass(color: body),
                      shadows: showcaseShadows(context, body, elevation: 5),
                      child: const SizedBox.expand(),
                    ),
                  ),
                  Positioned(
                    left: level + _handleGap / 2 - 2,
                    top: 8,
                    bottom: 8,
                    width: 4,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: body.withValues(alpha: 1),
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
                      shadows: showcaseShadows(
                        context,
                        frostedWhite,
                        elevation: 3,
                      ),
                      child: Align(
                        alignment: Alignment.centerRight,
                        child: Padding(
                          padding: const EdgeInsets.only(right: 18),
                          child: Icon(
                            Icons.brightness_6_rounded,
                            color: inkFor(darkWallpaper: darkWallpaper),
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
