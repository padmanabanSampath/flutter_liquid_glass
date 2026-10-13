import 'dart:math' as math;

import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/living_glass.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// Room the floating palette and style buttons take in the top trailing
/// corner, which page titles keep clear of.
const controlsClearance = 150.0;

/// Room the pebble tab bar takes at the bottom.
const tabBarClearance = 132.0;

/// A scrolling showcase page: a large title over [children], centered and
/// kept clear of the floating controls and the tab bar.
///
/// The glass of a page lives in the layers its children declare; each
/// layer samples the backdrop once.
class ShowcasePageFrame extends StatelessWidget {
  const ShowcasePageFrame({
    required this.title,
    required this.subtitle,
    required this.children,
    this.maxWidth = 520,
    this.ink,
    super.key,
  });

  final String title;
  final String subtitle;
  final List<Widget> children;
  final double maxWidth;

  /// Ink for the title; the showcase's by default.
  final Color? ink;

  @override
  Widget build(BuildContext context) {
    final ink = this.ink ?? ShowcaseScope.of(context).ink;
    return SafeArea(
      bottom: false,
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, tabBarClearance),
        child: Center(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxWidth: maxWidth),
            // Text on the wallpaper keeps a halo, so it reads over photos;
            // text on colored glass sets its own.
            child: DefaultTextStyle.merge(
              style: TextStyle(shadows: haloFor(ink)),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(
                      right: controlsClearance - 20,
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          title,
                          style: TextStyle(
                            color: ink,
                            fontSize: 34,
                            height: 1.1,
                            fontWeight: FontWeight.w700,
                            letterSpacing: -.6,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          subtitle,
                          style: TextStyle(
                            color: ink.withValues(alpha: .7),
                            fontSize: 15,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 26),
                  ...children,
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// The glass layer of a page section, in the showcase's current finish or
/// in [settings].
class ShowcaseLayer extends StatelessWidget {
  const ShowcaseLayer({required this.child, this.settings, super.key});

  final Widget child;

  /// Settings of this layer alone; the showcase's by default.
  final LiquidGlassSettings? settings;

  @override
  Widget build(BuildContext context) => LiquidGlassLayer(
    settings: settings ?? ShowcaseScope.of(context).settings,
    defaultAppearance: ShowcaseScope.of(context).frost,
    child: child,
  );
}

/// A card of sea glass: frosted white, or colored glass of [body].
class GlassCard extends StatelessWidget {
  const GlassCard({
    required this.child,
    this.padding = const EdgeInsets.all(18),
    this.radius = 28,
    this.body,
    this.elevation = 4,
    this.grouped = false,
    super.key,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;
  final Color? body;
  final double elevation;

  /// Whether the card blends with its siblings in a blend group.
  final bool grouped;

  @override
  Widget build(BuildContext context) {
    final body = this.body;
    final shape = LiquidRoundedSuperellipse(borderRadius: radius);
    final appearance = body == null
        ? null
        : LiquidGlassAppearance.seaGlass(color: body);
    final shadows = SeaGlassFinish.shadows(
      body ?? frostedWhite,
      elevation: elevation,
    );
    final content = Padding(padding: padding, child: child);
    return grouped
        ? LiquidGlass.grouped(
            shape: shape,
            appearance: appearance,
            shadows: shadows,
            child: content,
          )
        : LiquidGlass(
            shape: shape,
            appearance: appearance,
            shadows: shadows,
            child: content,
          );
  }
}

/// A round sea glass button with an icon, frosted or of [body]. Its outline
/// follows the showcase's shape: an oval, a tumbled pebble of its own or a
/// living pebble.
class GlassPebble extends StatelessWidget {
  const GlassPebble({
    required this.icon,
    required this.label,
    this.size = 56,
    this.iconSize,
    this.body,
    this.onTap,
    this.grouped = false,
    super.key,
  });

  final IconData icon;
  final String label;
  final double size;
  final double? iconSize;
  final Color? body;
  final VoidCallback? onTap;
  final bool grouped;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final body = this.body;
    final ink = body == null ? style.ink : inkOn(body);
    final content = Center(
      child: Icon(icon, color: ink, size: iconSize ?? size * .42),
    );
    final shape = pebbleOutline(context, style.shape, seedOf(label));
    final appearance = body == null
        ? null
        : LiquidGlassAppearance.seaGlass(color: body);
    final shadows = SeaGlassFinish.shadows(
      body ?? frostedWhite,
      elevation: size > 64 ? 7 : 4,
    );
    return Semantics(
      button: true,
      label: label,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: SizedBox.square(
          dimension: size,
          child: LiquidStretch.seaGlass(
            child: grouped
                ? LiquidGlass.grouped(
                    shape: shape,
                    appearance: appearance,
                    shadows: shadows,
                    child: content,
                  )
                : LiquidGlass(
                    shape: shape,
                    appearance: appearance,
                    shadows: shadows,
                    child: content,
                  ),
          ),
        ),
      ),
    );
  }
}

/// A slider of two glass bars: colored glass for the level and frosted
/// glass for the rest, split by a thin handle. Two shapes.
class GlassSlider extends StatelessWidget {
  const GlassSlider({
    required this.value,
    required this.onChanged,
    required this.label,
    required this.trailing,
    this.height = 52,
    super.key,
  });

  final double value;
  final ValueChanged<double> onChanged;
  final String label;
  final IconData trailing;
  final double height;

  static const _handleGap = 10.0;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    final body = style.body;
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final travel = width - 2 * height - _handleGap;
        final level = height + travel * value;
        void update(Offset local) =>
            onChanged(((local.dx - height) / travel).clamp(0.0, 1.0));
        return Semantics(
          slider: true,
          label: label,
          value: '${(value * 100).round()}%',
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onHorizontalDragDown: (details) => update(details.localPosition),
            onHorizontalDragUpdate: (details) => update(details.localPosition),
            child: SizedBox(
              height: height,
              child: Stack(
                children: [
                  Positioned(
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: level,
                    child: LiquidGlass(
                      shape: LiquidRoundedSuperellipse(
                        borderRadius: height / 2,
                      ),
                      appearance: LiquidGlassAppearance.seaGlass(color: body),
                      shadows: SeaGlassFinish.shadows(body, elevation: 4),
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
                      shape: LiquidRoundedSuperellipse(
                        borderRadius: height / 2,
                      ),
                      shadows: SeaGlassFinish.shadows(
                        frostedWhite,
                        elevation: 3,
                      ),
                      child: Align(
                        alignment: Alignment.centerRight,
                        child: Padding(
                          padding: const EdgeInsets.only(right: 16),
                          child: Icon(trailing, color: style.ink, size: 22),
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

/// A small uppercase heading above a section.
class SectionHeader extends StatelessWidget {
  const SectionHeader(this.text, {this.trailing, this.ink, super.key});

  final String text;
  final String? trailing;

  /// Ink for the heading; the showcase's by default.
  final Color? ink;

  @override
  Widget build(BuildContext context) {
    final ink = this.ink ?? ShowcaseScope.of(context).ink;
    final style = TextStyle(
      color: ink.withValues(alpha: .72),
      fontSize: 12,
      fontWeight: FontWeight.w700,
      letterSpacing: 1.3,
    );
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 0, 4, 12),
      child: Row(
        children: [
          Text(text.toUpperCase(), style: style),
          const Spacer(),
          if (trailing case final trailing?)
            Text(trailing, style: style.copyWith(letterSpacing: 0)),
        ],
      ),
    );
  }
}

/// A soft disc behind an icon on frosted glass.
class IconBubble extends StatelessWidget {
  const IconBubble({
    required this.icon,
    this.size = 40,
    this.color,
    this.iconColor,
    super.key,
  });

  final IconData icon;
  final double size;

  /// The disc color; a soft white by default.
  final Color? color;
  final Color? iconColor;

  @override
  Widget build(BuildContext context) {
    final style = ShowcaseScope.of(context);
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color ?? Color(style.dark ? 0x24FFFFFF : 0x4DFFFFFF),
      ),
      child: Icon(icon, color: iconColor ?? style.ink, size: size * .5),
    );
  }
}

/// A labelled slider whose knob is a pebble of sea glass of [knob] on a
/// painted track. Place it in a [LiquidGlassLayer]; the knob is one shape.
class GlassValueSlider extends StatelessWidget {
  const GlassValueSlider({
    required this.label,
    required this.value,
    required this.onChanged,
    required this.knob,
    required this.ink,
    super.key,
  });

  final String label;

  /// From `0` to `1`.
  final double value;
  final ValueChanged<double> onChanged;
  final Color knob;
  final Color ink;

  static const _knob = 30.0;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 104,
          child: Text(
            label,
            style: TextStyle(
              color: ink,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              final travel = math.max<double>(constraints.maxWidth - _knob, 1);
              final x = travel * value.clamp(0.0, 1.0);
              void update(Offset local) =>
                  onChanged(((local.dx - _knob / 2) / travel).clamp(0.0, 1.0));
              return Semantics(
                slider: true,
                label: label,
                value: '${(value * 100).round()}%',
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onHorizontalDragDown: (details) =>
                      update(details.localPosition),
                  onHorizontalDragUpdate: (details) =>
                      update(details.localPosition),
                  child: SizedBox(
                    height: 44,
                    child: Stack(
                      children: [
                        Positioned(
                          left: _knob / 2,
                          right: _knob / 2,
                          top: 19,
                          height: 6,
                          child: DecoratedBox(
                            decoration: BoxDecoration(
                              color: ink.withValues(alpha: .14),
                              borderRadius: BorderRadius.circular(3),
                            ),
                          ),
                        ),
                        Positioned(
                          left: _knob / 2,
                          width: x,
                          top: 19,
                          height: 6,
                          child: DecoratedBox(
                            decoration: BoxDecoration(
                              color: ink.withValues(alpha: .42),
                              borderRadius: BorderRadius.circular(3),
                            ),
                          ),
                        ),
                        Positioned(
                          left: x,
                          top: 7,
                          width: _knob,
                          height: _knob,
                          child: LiquidGlass(
                            shape: const LiquidOval(),
                            appearance: LiquidGlassAppearance.seaGlass(
                              color: knob,
                            ),
                            child: const SizedBox.expand(),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
