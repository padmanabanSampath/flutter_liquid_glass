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
    this.shape = ShowcaseShape.round,
    this.shine = ShowcaseShine.satin,
    this.lightAngle = 0,
    this.clarity = defaultClarity,
    this.colorStrength = defaultColorStrength,
    this.material = ShowcaseMaterial.seaGlass,
  });

  /// Index into [seaGlassPalette]; teal by default.
  final int accentIndex;
  final ShowcaseTone tone;
  final ShowcaseFinish finish;
  final ShowcaseWallpaper wallpaper;

  /// How bright the wallpaper is, from 0 (dimmed) to 1.
  final double brightness;

  /// The outline of the round buttons.
  final ShowcaseShape shape;

  /// How the glass catches the light.
  final ShowcaseShine shine;

  /// Where the light comes from, in radians clockwise from straight above;
  /// the sun of the tide pool sets it.
  final double lightAngle;

  /// How clear colored glass is, from `0` (as tumbled) to `1`; see
  /// [ShowcaseFinish.settings].
  final double clarity;

  /// How strong the color of colored glass is; see [seaGlassBody].
  final double colorStrength;

  /// Which glass every page is made of.
  final ShowcaseMaterial material;

  /// The appearance of frosted glass: switched-off tiles and panels.
  LiquidGlassAppearance get frost =>
      LiquidGlassAppearance.seaGlass(color: material.frost);

  String get accentName => seaGlassPalette[accentIndex].name;
  Color get accent => seaGlassPalette[accentIndex].color;

  /// The body of colored glass: [accent] in [tone].
  Color get body =>
      tone.body(accent, strength: colorStrength, material: material);

  /// The body of [color] in this style's [tone].
  Color bodyOf(Color color) =>
      tone.body(color, strength: colorStrength, material: material);

  /// The settings of every glass layer: the finish, shine and light of our
  /// sea glass, or slab glass under the same light. Transparency clears
  /// colored glass, slab glass only a little.
  LiquidGlassSettings get settings => switch (material) {
    ShowcaseMaterial.seaGlass => finish.settings(
      shine: shine,
      lightAngle: lightAngle,
      clarity: clarity,
    ),
    ShowcaseMaterial.slab => LiquidGlassSettings.slabSeaGlass(
      finish: slabFinish.copyWith(
        colorClarity: clarity * material.clarityShare,
      ),
      lightAngle: lightAngle,
    ),
  };

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
    ShowcaseShape? shape,
    ShowcaseShine? shine,
    double? lightAngle,
    double? clarity,
    double? colorStrength,
    ShowcaseMaterial? material,
  }) => ShowcaseStyle(
    accentIndex: accentIndex ?? this.accentIndex,
    tone: tone ?? this.tone,
    finish: finish ?? this.finish,
    wallpaper: wallpaper ?? this.wallpaper,
    brightness: brightness ?? this.brightness,
    shape: shape ?? this.shape,
    shine: shine ?? this.shine,
    lightAngle: lightAngle ?? this.lightAngle,
    clarity: clarity ?? this.clarity,
    colorStrength: colorStrength ?? this.colorStrength,
    material: material ?? this.material,
  );

  @override
  bool operator ==(Object other) =>
      other is ShowcaseStyle &&
      other.accentIndex == accentIndex &&
      other.tone == tone &&
      other.finish == finish &&
      other.wallpaper == wallpaper &&
      other.brightness == brightness &&
      other.shape == shape &&
      other.shine == shine &&
      other.lightAngle == lightAngle &&
      other.clarity == clarity &&
      other.colorStrength == colorStrength &&
      other.material == material;

  @override
  int get hashCode => Object.hash(
    accentIndex,
    tone,
    finish,
    wallpaper,
    brightness,
    shape,
    shine,
    lightAngle,
    clarity,
    colorStrength,
    material,
  );
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
