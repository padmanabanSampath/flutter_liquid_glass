import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// One destination of a [PebbleTabBar].
typedef PebbleTab = ({IconData icon, String label});

/// A tab bar strung from sea glass pebbles.
///
/// Every tab is a pebble of one blend group, close enough that neighbors
/// join through a thin neck of glass. The selected tab swells into a
/// capsule of the showcase's body color that shows its label, and the glass
/// flows as the others make room. The pebbles shrink to fit narrow screens.
/// Its own layer: [tabs] shapes.
class PebbleTabBar extends StatelessWidget {
  const PebbleTabBar({
    required this.tabs,
    required this.selected,
    required this.onSelected,
    super.key,
  });

  final List<PebbleTab> tabs;
  final int selected;
  final ValueChanged<int> onSelected;

  static const height = 56.0;
  static const _margin = 16.0;
  static const _gap = 6.0;
  static const _labelWidth = 66.0;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    return LayoutBuilder(
      builder: (context, constraints) {
        final room =
            constraints.maxWidth -
            2 * _margin -
            _labelWidth -
            _gap * (tabs.length - 1);
        final size = (room / tabs.length).clamp(40.0, height);
        return LiquidGlassLayer(
          settings: style.settings,
          defaultAppearance: style.frost,
          child: LiquidGlassBlendGroup(
            blend: 14,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                for (var i = 0; i < tabs.length; i++) ...[
                  if (i > 0) const SizedBox(width: _gap),
                  _Pebble(
                    tab: tabs[i],
                    size: size,
                    selected: i == selected,
                    style: style,
                    onTap: () => onSelected(i),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

class _Pebble extends StatelessWidget {
  const _Pebble({
    required this.tab,
    required this.size,
    required this.selected,
    required this.style,
    required this.onTap,
  });

  final PebbleTab tab;
  final double size;
  final bool selected;
  final ShowcaseStyle style;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final body = style.body;
    final ink = selected ? inkOn(body) : style.ink;
    return Semantics(
      button: true,
      selected: selected,
      label: tab.label,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 520),
          curve: Curves.easeOutBack,
          width: size + (selected ? PebbleTabBar._labelWidth : 0),
          height: size,
          child: LiquidStretch.seaGlass(
            child: LiquidGlass.grouped(
              shape: LiquidRoundedSuperellipse(
                borderRadius: size / 2,
              ),
              appearance: selected
                  ? LiquidGlassAppearance.seaGlass(color: body)
                  : null,
              shadows: SeaGlassFinish.shadows(
                selected ? body : frostedWhite,
                elevation: selected ? 6 : 3,
              ),
              child: ClipRect(
                // Without minWidth 0 the pebble's width would be the row's
                // minimum, and the row would start its icon at the left.
                child: OverflowBox(
                  minWidth: 0,
                  maxWidth: double.infinity,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(tab.icon, color: ink, size: size * .43),
                      AnimatedSize(
                        duration: const Duration(milliseconds: 360),
                        curve: Curves.easeOutCubic,
                        child: selected
                            ? Padding(
                                padding: const EdgeInsets.only(left: 8),
                                child: Text(
                                  tab.label,
                                  maxLines: 1,
                                  style: TextStyle(
                                    color: ink,
                                    fontSize: 14,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              )
                            : const SizedBox.shrink(),
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
