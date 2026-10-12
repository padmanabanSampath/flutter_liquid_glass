import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/wallpapers.dart';

/// The sea glass every page of the showcase is made of: the color, how
/// tumbled and how dense it is, and what it lies on.
@immutable
class ShowcaseStyle {
  const ShowcaseStyle({
    this.accentIndex = 2,
    this.tone = ShowcaseTone.vivid,
    this.finish = ShowcaseFinish.tumbled,
    this.wallpaper = ShowcaseWallpaper.seaMist,
    this.brightness = .7,
  });

  /// Index into [seaGlassPalette]; teal by default.
  final int accentIndex;
  final ShowcaseTone tone;
  final ShowcaseFinish finish;
  final ShowcaseWallpaper wallpaper;

  /// How bright the wallpaper is, from 0 (dimmed) to 1.
  final double brightness;

  String get accentName => seaGlassPalette[accentIndex].name;
  Color get accent => seaGlassPalette[accentIndex].color;

  /// The body of colored glass: [accent] in [tone].
  Color get body => tone.body(accent);

  /// The body of [color] in this style's [tone].
  Color bodyOf(Color color) => tone.body(color);

  LiquidGlassSettings get settings => finish.settings;

  /// Whether the wallpaper needs light ink.
  bool get dark => wallpaper.dark;

  /// Ink for frosted glass and text on the wallpaper.
  Color get ink => inkFor(darkWallpaper: dark);

  /// Ink for glass of the body color.
  Color get bodyInk => inkOn(body);

  ShowcaseStyle copyWith({
    int? accentIndex,
    ShowcaseTone? tone,
    ShowcaseFinish? finish,
    ShowcaseWallpaper? wallpaper,
    double? brightness,
  }) => ShowcaseStyle(
    accentIndex: accentIndex ?? this.accentIndex,
    tone: tone ?? this.tone,
    finish: finish ?? this.finish,
    wallpaper: wallpaper ?? this.wallpaper,
    brightness: brightness ?? this.brightness,
  );

  @override
  bool operator ==(Object other) =>
      other is ShowcaseStyle &&
      other.accentIndex == accentIndex &&
      other.tone == tone &&
      other.finish == finish &&
      other.wallpaper == wallpaper &&
      other.brightness == brightness;

  @override
  int get hashCode =>
      Object.hash(accentIndex, tone, finish, wallpaper, brightness);
}

/// Hands the showcase's [ShowcaseStyle] to every page.
class ShowcaseScope extends InheritedNotifier<ValueNotifier<ShowcaseStyle>> {
  const ShowcaseScope({
    required ValueNotifier<ShowcaseStyle> style,
    required super.child,
    super.key,
  }) : super(notifier: style);

  /// The current style; rebuilds [context] when it changes.
  static ShowcaseStyle of(BuildContext context) => controllerOf(context).value;

  /// The notifier that holds the style, to change it.
  static ValueNotifier<ShowcaseStyle> controllerOf(BuildContext context) {
    final scope = context.dependOnInheritedWidgetOfExactType<ShowcaseScope>();
    assert(scope != null, 'No ShowcaseScope above this widget.');
    return scope!.notifier!;
  }

  /// Applies [change] to the current style.
  static void update(
    BuildContext context,
    ShowcaseStyle Function(ShowcaseStyle style) change,
  ) {
    final controller = context
        .getInheritedWidgetOfExactType<ShowcaseScope>()!
        .notifier!;
    controller.value = change(controller.value);
  }
}
