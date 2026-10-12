import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Which Impeller backend the app asks for at launch, on Android.
enum RendererChoice {
  /// The device rule in the Android app: OpenGL ES where it measured
  /// leaner, the engine's default elsewhere.
  auto('Auto'),
  vulkan('Vulkan'),
  opengles('OpenGL ES');

  const RendererChoice(this.label);

  final String label;
}

/// The renderer choice and what this launch runs on.
@immutable
class RendererState {
  const RendererState({
    required this.choice,
    required this.launched,
    required this.auto,
  });

  final RendererChoice choice;

  /// The backend this launch requested: `vulkan`, `opengles`, or `default`.
  final String launched;

  /// What [RendererChoice.auto] requests on this device.
  final String auto;

  /// The backend [choice] requests at the next launch.
  String get requested => choice == RendererChoice.auto ? auto : choice.name;

  /// Whether [choice] only takes effect after a restart.
  bool get needsRestart => requested != launched;

  RendererState copyWith({RendererChoice? choice}) => RendererState(
    choice: choice ?? this.choice,
    launched: launched,
    auto: auto,
  );

  /// A label for a requested backend.
  static String describe(String backend) => switch (backend) {
    'vulkan' => 'Vulkan',
    'opengles' => 'OpenGL ES',
    _ => 'Vulkan (engine default)',
  };
}

/// Talks to the Android app, which picks the backend before the engine
/// starts: the engine reads it once per process, so changes apply after
/// [restart].
abstract final class RendererBackend {
  static const _channel = MethodChannel('dev.liquid_glass_renderer/renderer');

  /// Whether this platform lets the app pick a backend.
  static bool get supported =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  /// The current state, or `null` where the app cannot pick a backend.
  static Future<RendererState?> load() async {
    if (!supported) return null;
    try {
      final state = await _channel.invokeMapMethod<String, Object?>('get');
      if (state == null) return null;
      return RendererState(
        choice: RendererChoice.values.firstWhere(
          (choice) => choice.name == state['choice'],
          orElse: () => RendererChoice.auto,
        ),
        launched: state['launched'] as String? ?? 'default',
        auto: state['auto'] as String? ?? 'default',
      );
    } on MissingPluginException {
      return null;
    } on PlatformException {
      return null;
    }
  }

  /// Saves [choice] for the next launch.
  static Future<void> choose(RendererChoice choice) =>
      _channel.invokeMethod<void>('set', choice.name);

  /// Restarts the app so the engine starts on the chosen backend.
  static Future<void> restart() => _channel.invokeMethod<void>('restart');
}
