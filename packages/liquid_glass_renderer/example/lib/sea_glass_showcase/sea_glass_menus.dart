import 'dart:math' as math;

import 'package:flutter/material.dart' show Icons;
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/renderer_backend.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/wallpapers.dart';

/// The floating controls of the showcase, in the bottom trailing corner, or
/// the top trailing one with [top].
///
/// The palette button bursts into every sea glass color: the pebbles start
/// inside it and pull out of its glass, blending with it on the way, then
/// settle on two arcs. The style button opens a tray with the wallpaper,
/// the finish and the tone. All of it is one layer: at most fifteen shapes
/// are on screen at once, below the sixteen a layer can render.
class SeaGlassControls extends StatefulWidget {
  const SeaGlassControls({
    required this.accentIndex,
    required this.tone,
    required this.finish,
    required this.wallpaper,
    required this.onAccent,
    required this.onTone,
    required this.onFinish,
    required this.onWallpaper,
    this.top = false,
    super.key,
  });

  /// Whether the controls sit in the top trailing corner; the palette then
  /// opens downward and the tray drops below the buttons.
  final bool top;
  final int accentIndex;
  final ShowcaseTone tone;
  final ShowcaseFinish finish;
  final ShowcaseWallpaper wallpaper;
  final ValueChanged<int> onAccent;
  final ValueChanged<ShowcaseTone> onTone;
  final ValueChanged<ShowcaseFinish> onFinish;
  final ValueChanged<ShowcaseWallpaper> onWallpaper;

  @override
  State<SeaGlassControls> createState() => _SeaGlassControlsState();
}

class _SeaGlassControlsState extends State<SeaGlassControls>
    with TickerProviderStateMixin {
  late final AnimationController _palette = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 720),
  );
  late final AnimationController _style = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 420),
  );

  static const _fabSize = 60.0;
  static const _buttonSize = 52.0;
  static const _buttonGap = 14.0;
  static const _margin = 24.0;
  static const _topMargin = 14.0;
  static const _blendAtRest = 8.0;
  static const _blendInFlight = 30.0;

  /// The palette as two arcs from the left of the button to above it: the
  /// first colors on the inner arc, the rest on the outer one.
  static const _innerCount = 4;
  static const _innerRadius = 88.0;
  static const _innerSize = 44.0;
  static const _outerRadius = 152.0;
  static const _outerSize = 38.0;

  @override
  void dispose() {
    _palette.dispose();
    _style.dispose();
    super.dispose();
  }

  bool _isOpen(AnimationController controller) =>
      controller.status == AnimationStatus.forward ||
      controller.status == AnimationStatus.completed;

  void _togglePalette() {
    if (_isOpen(_palette)) {
      _palette.reverse();
    } else {
      _style.reverse();
      _palette.forward();
    }
  }

  void _toggleStyle() {
    if (_isOpen(_style)) {
      _style.reverse();
    } else {
      _palette.reverse();
      _style.forward();
    }
  }

  void _closeAll() {
    _palette.reverse();
    _style.reverse();
  }

  /// Where pebble [index] settles, relative to the palette button's center,
  /// and its size.
  static (Offset, double) _pebbleSlot(int index, {required bool top}) {
    final inner = index < _innerCount;
    final count = inner ? _innerCount : seaGlassPalette.length - _innerCount;
    final slot = inner ? index : index - _innerCount;
    // From pointing left (180°) to pointing up (270°), or down (90°) when
    // the controls sit at the top.
    final sweep = .5 * slot / (count - 1);
    final angle = math.pi * (top ? 1 - sweep : 1 + sweep);
    final radius = inner ? _innerRadius : _outerRadius;
    return (
      Offset(math.cos(angle), math.sin(angle)) * radius,
      inner ? _innerSize : _outerSize,
    );
  }

  @override
  Widget build(BuildContext context) {
    final ink = inkFor(darkWallpaper: widget.wallpaper.dark);
    final accent = seaGlassPalette[widget.accentIndex];
    final body = widget.tone.body(accent.color);
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = constraints.biggest;
        final safe = MediaQuery.paddingOf(context);
        final top = widget.top;
        final fabCenter = Offset(
          size.width - (top ? _margin - 6 : _margin) - _fabSize / 2,
          top
              ? safe.top + _topMargin + _fabSize / 2
              : size.height - safe.bottom - _margin - _fabSize / 2,
        );
        final buttonCenter =
            fabCenter -
            const Offset(_fabSize / 2 + _buttonGap + _buttonSize / 2, 0);
        return AnimatedBuilder(
          animation: Listenable.merge([_palette, _style]),
          builder: (context, _) {
            final open = _palette.value;
            final styleOpen = _style.value;
            return Stack(
              children: [
                if (open > 0 || styleOpen > 0)
                  Positioned.fill(
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onTap: _closeAll,
                    ),
                  ),
                if (styleOpen > 0)
                  Positioned(
                    right: 16,
                    top: top ? safe.top + _topMargin + _fabSize + 14 : null,
                    bottom: top ? null : safe.bottom + _margin + _fabSize + 16,
                    width: math.min(size.width - 32, 420),
                    child: _StyleTray(
                      fromTop: top,
                      progress: Curves.easeOutCubic.transform(styleOpen),
                      ink: ink,
                      accent: accent.color,
                      body: body,
                      wallpaper: widget.wallpaper,
                      finish: widget.finish,
                      tone: widget.tone,
                      onWallpaper: widget.onWallpaper,
                      onFinish: widget.onFinish,
                      onTone: widget.onTone,
                    ),
                  ),
                if (open > 0)
                  Positioned(
                    right:
                        size.width -
                        (fabCenter.dx - _outerRadius - _outerSize / 2 - 14),
                    top: fabCenter.dy - 12,
                    child: Opacity(
                      opacity: Curves.easeIn.transform(open),
                      child: Text(
                        accent.name,
                        style: TextStyle(
                          color: ink,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),
                Positioned.fill(
                  child: LiquidGlassBlendGroup(
                    blend:
                        _blendAtRest +
                        (_blendInFlight - _blendAtRest) *
                            math.sin(math.pi * open),
                    child: Stack(
                      children: [
                        if (open < 1)
                          _place(
                            Offset.lerp(
                              buttonCenter,
                              fabCenter,
                              Curves.easeInCubic.transform(open),
                            )!,
                            _buttonSize * (1 - open),
                            _RoundButton(
                              label: 'Style',
                              icon: Icons.wallpaper_rounded,
                              ink: ink,
                              onTap: _toggleStyle,
                            ),
                          ),
                        if (open > 0)
                          for (var i = 0; i < seaGlassPalette.length; i++)
                            _pebble(i, open, fabCenter),
                        _place(
                          fabCenter,
                          _fabSize,
                          _RoundButton(
                            label: open > .5 ? 'Close palette' : 'Palette',
                            icon: open > .5
                                ? Icons.close_rounded
                                : Icons.palette_rounded,
                            ink: inkOn(body),
                            body: body,
                            turns: open * .25,
                            onTap: _togglePalette,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }

  Widget _pebble(int index, double open, Offset fabCenter) {
    final count = seaGlassPalette.length;
    final start = .3 * index / count;
    final local = ((open - start) / .7).clamp(0.0, 1.0);
    final (slot, size) = _pebbleSlot(index, top: widget.top);
    final color = widget.tone.body(seaGlassPalette[index].color);
    return _place(
      fabCenter + slot * Curves.easeOutBack.transform(local),
      14 + (size - 14) * Curves.easeOutCubic.transform(local),
      _Pebble(
        name: seaGlassPalette[index].name,
        body: color,
        selected: index == widget.accentIndex,
        onTap: () => widget.onAccent(index),
      ),
    );
  }

  static Widget _place(Offset center, double size, Widget child) => Positioned(
    left: center.dx - size / 2,
    top: center.dy - size / 2,
    width: size,
    height: size,
    child: child,
  );
}

/// A round sea glass button in the blend group.
class _RoundButton extends StatelessWidget {
  const _RoundButton({
    required this.label,
    required this.icon,
    required this.ink,
    required this.onTap,
    this.body,
    this.turns = 0,
  });

  final String label;
  final IconData icon;
  final Color ink;
  final VoidCallback onTap;

  /// The body color; frosted white glass without one.
  final Color? body;
  final double turns;

  @override
  Widget build(BuildContext context) {
    final body = this.body;
    return Semantics(
      button: true,
      label: label,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: LiquidStretch.seaGlass(
          child: LiquidGlass.grouped(
            shape: const LiquidOval(),
            appearance: body == null
                ? null
                : LiquidGlassAppearance.seaGlass(color: body),
            shadows: SeaGlassFinish.shadows(body ?? frostedWhite),
            child: Center(
              child: Transform.rotate(
                angle: turns * 2 * math.pi,
                child: FittedBox(child: Icon(icon, color: ink, size: 26)),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// One color of the palette menu: a small sea glass pebble.
class _Pebble extends StatelessWidget {
  const _Pebble({
    required this.name,
    required this.body,
    required this.selected,
    required this.onTap,
  });

  final String name;
  final Color body;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      selected: selected,
      label: name,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: LiquidStretch.seaGlass(
          child: LiquidGlass.grouped(
            shape: const LiquidOval(),
            appearance: LiquidGlassAppearance.seaGlass(color: body),
            shadows: SeaGlassFinish.shadows(body, elevation: 4),
            // Small pebbles are thin, so over a saturated wallpaper they
            // show mostly the backdrop. A solid core keeps each color
            // recognizable.
            child: Center(
              child: FractionallySizedBox(
                widthFactor: selected ? .5 : .3,
                heightFactor: selected ? .5 : .3,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: body.withValues(alpha: 1),
                    border: Border.all(
                      color: const Color(0xCCFFFFFF),
                      width: 1.5,
                    ),
                  ),
                  child: selected
                      ? FittedBox(
                          child: Icon(
                            Icons.check_rounded,
                            color: inkOn(body.withValues(alpha: 1)),
                          ),
                        )
                      : null,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// A frosted tray with the wallpaper, finish and tone.
///
/// The selected finish and tone sit on pebbles of the chosen sea glass.
/// Shapes in one layer merge rather than stack, so the pebbles have a layer
/// of their own inside the tray, which samples the tray's glass beneath them.
class _StyleTray extends StatelessWidget {
  const _StyleTray({
    required this.fromTop,
    required this.progress,
    required this.ink,
    required this.accent,
    required this.body,
    required this.wallpaper,
    required this.finish,
    required this.tone,
    required this.onWallpaper,
    required this.onFinish,
    required this.onTone,
  });

  /// Whether the tray drops from buttons above it.
  final bool fromTop;
  final double progress;
  final Color ink;
  final Color accent;

  /// The sea glass body of the selection pebbles.
  final Color body;
  final ShowcaseWallpaper wallpaper;
  final ShowcaseFinish finish;
  final ShowcaseTone tone;
  final ValueChanged<ShowcaseWallpaper> onWallpaper;
  final ValueChanged<ShowcaseFinish> onFinish;
  final ValueChanged<ShowcaseTone> onTone;

  @override
  Widget build(BuildContext context) {
    return Transform.translate(
      offset: Offset(0, (fromTop ? -28 : 28) * (1 - progress)),
      child: LiquidGlassVisibility(
        visibility: progress,
        child: LiquidGlass(
          shape: const LiquidRoundedSuperellipse(borderRadius: 30),
          shadows: SeaGlassFinish.shadows(frostedWhite),
          child: Opacity(
            opacity: progress,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(18, 16, 18, 18),
              child: LiquidGlassLayer(
                settings: LiquidGlassSettings.of(context),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _SectionLabel('Wallpaper', ink),
                    SizedBox(
                      height: 86,
                      child: ListView.separated(
                        scrollDirection: Axis.horizontal,
                        itemCount: ShowcaseWallpaper.values.length,
                        separatorBuilder: (_, _) => const SizedBox(width: 8),
                        itemBuilder: (context, index) {
                          final option = ShowcaseWallpaper.values[index];
                          return _WallpaperChip(
                            wallpaper: option,
                            accent: accent,
                            ink: ink,
                            selected: option == wallpaper,
                            onTap: () => onWallpaper(option),
                          );
                        },
                      ),
                    ),
                    const SizedBox(height: 10),
                    _SectionLabel('Finish', ink),
                    _Segmented(
                      labels: [for (final f in ShowcaseFinish.values) f.label],
                      selected: finish.index,
                      ink: ink,
                      body: body,
                      onSelected: (i) => onFinish(ShowcaseFinish.values[i]),
                    ),
                    const SizedBox(height: 14),
                    _SectionLabel('Tone', ink),
                    _Segmented(
                      labels: [for (final t in ShowcaseTone.values) t.label],
                      selected: tone.index,
                      ink: ink,
                      body: body,
                      onSelected: (i) => onTone(ShowcaseTone.values[i]),
                    ),
                    if (RendererBackend.supported)
                      _RendererSection(ink: ink, body: body),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// The Impeller backend the Android app requests at launch. The choice is
/// saved at once and applies after a restart, which the engine needs to
/// read it again.
class _RendererSection extends StatefulWidget {
  const _RendererSection({required this.ink, required this.body});

  final Color ink;
  final Color body;

  @override
  State<_RendererSection> createState() => _RendererSectionState();
}

class _RendererSectionState extends State<_RendererSection> {
  RendererState? _state;

  @override
  void initState() {
    super.initState();
    RendererBackend.load().then((state) {
      if (mounted) setState(() => _state = state);
    });
  }

  Future<void> _choose(RendererChoice choice) async {
    await RendererBackend.choose(choice);
    if (mounted) setState(() => _state = _state?.copyWith(choice: choice));
  }

  @override
  Widget build(BuildContext context) {
    final state = _state;
    if (state == null) return const SizedBox.shrink();
    final ink = widget.ink;
    final body = widget.body;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 14),
        _SectionLabel('Renderer', ink),
        _Segmented(
          labels: [for (final choice in RendererChoice.values) choice.label],
          selected: state.choice.index,
          ink: ink,
          body: body,
          onSelected: (i) => _choose(RendererChoice.values[i]),
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: Text(
                state.needsRestart
                    ? 'Running on ${RendererState.describe(state.launched)}. '
                          'Restart for '
                          '${RendererState.describe(state.requested)}.'
                    : 'Running on ${RendererState.describe(state.launched)}'
                          '${state.choice == RendererChoice.auto ? ', '
                                    'picked for this device' : ''}.',
                style: TextStyle(
                  color: ink.withValues(alpha: .75),
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            if (state.needsRestart) ...[
              const SizedBox(width: 10),
              Semantics(
                button: true,
                label: 'Restart to apply',
                excludeSemantics: true,
                child: GestureDetector(
                  onTap: RendererBackend.restart,
                  child: LiquidStretch.seaGlass(
                    child: LiquidGlass(
                      shape: const LiquidRoundedSuperellipse(borderRadius: 17),
                      appearance: LiquidGlassAppearance.seaGlass(color: body),
                      shadows: SeaGlassFinish.shadows(body, elevation: 3),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 9,
                        ),
                        child: Text(
                          'Restart',
                          style: TextStyle(
                            color: inkOn(body),
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ],
        ),
      ],
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text, this.ink);

  final String text;
  final Color ink;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        text.toUpperCase(),
        style: TextStyle(
          color: ink.withValues(alpha: .7),
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.2,
        ),
      ),
    );
  }
}

class _WallpaperChip extends StatelessWidget {
  const _WallpaperChip({
    required this.wallpaper,
    required this.accent,
    required this.ink,
    required this.selected,
    required this.onTap,
  });

  final ShowcaseWallpaper wallpaper;
  final Color accent;
  final Color ink;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      selected: selected,
      label: wallpaper.label,
      excludeSemantics: true,
      child: GestureDetector(
        onTap: onTap,
        child: SizedBox(
          width: 64,
          child: Column(
            children: [
              AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: 56,
                height: 56,
                padding: const EdgeInsets.all(3),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: selected ? ink : ink.withValues(alpha: 0),
                    width: 2,
                  ),
                ),
                child: WallpaperThumbnail(wallpaper: wallpaper, accent: accent),
              ),
              const SizedBox(height: 6),
              Text(
                wallpaper.label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: ink,
                  fontSize: 11,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A segmented control whose selection is a sea glass pebble of [body]
/// that slides to the chosen segment.
class _Segmented extends StatelessWidget {
  const _Segmented({
    required this.labels,
    required this.selected,
    required this.ink,
    required this.body,
    required this.onSelected,
  });

  final List<String> labels;
  final int selected;
  final Color ink;
  final Color body;
  final ValueChanged<int> onSelected;

  static const _height = 40.0;
  static const _inset = 3.0;

  @override
  Widget build(BuildContext context) {
    final selectedInk = inkOn(body);
    return LayoutBuilder(
      builder: (context, constraints) {
        final segment = constraints.maxWidth / labels.length;
        return SizedBox(
          height: _height,
          child: Stack(
            children: [
              Positioned.fill(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: ink.withValues(alpha: .08),
                    borderRadius: const BorderRadius.all(
                      Radius.circular(_height / 2),
                    ),
                  ),
                ),
              ),
              AnimatedPositioned(
                duration: const Duration(milliseconds: 360),
                curve: Curves.easeOutCubic,
                left: segment * selected + _inset,
                top: _inset,
                bottom: _inset,
                width: segment - 2 * _inset,
                child: LiquidGlass(
                  shape: const LiquidRoundedSuperellipse(
                    borderRadius: _height / 2 - _inset,
                  ),
                  appearance: LiquidGlassAppearance.seaGlass(color: body),
                  shadows: SeaGlassFinish.shadows(body, elevation: 2),
                  child: const SizedBox.expand(),
                ),
              ),
              Row(
                children: [
                  for (var i = 0; i < labels.length; i++)
                    Expanded(
                      child: Semantics(
                        button: true,
                        selected: i == selected,
                        label: labels[i],
                        excludeSemantics: true,
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onTap: () => onSelected(i),
                          child: Center(
                            child: AnimatedDefaultTextStyle(
                              duration: const Duration(milliseconds: 200),
                              style: TextStyle(
                                color: i == selected ? selectedInk : ink,
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                              ),
                              child: Text(labels[i]),
                            ),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}
