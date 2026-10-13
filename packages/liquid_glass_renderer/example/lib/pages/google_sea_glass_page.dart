import 'dart:async';
import 'dart:math' as math;
import 'dart:ui' show lerpDouble;

import 'package:flutter/material.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';

/// A recreation of Google's Sea Glass concept: a sheet of Material
/// components in thick, milky pastel glass under a light that follows the
/// pointer.
///
/// The light shades every slab and casts its shadow
/// ([LiquidGlassSettings.googleSeaGlass]). The moon dims the room to dusk,
/// the flashlight turns the light into a beam aimed by the pointer, and the
/// slider dims everything. Tap G for the other glass sets.
///
/// The components share one glass layer; the floating action button, its
/// menu and the toasts float above in a second one, so neither passes a
/// layer's sixteen shapes.
class GoogleSeaGlassPage extends StatefulWidget {
  const GoogleSeaGlassPage({super.key});

  @override
  State<GoogleSeaGlassPage> createState() => GoogleSeaGlassPageState();
}

/// The two tracks the media card plays.
const _tracks = [
  (
    title: 'Sea Glass',
    artist: 'Tide Pool Radio',
    art: Icons.waves,
    length: 204,
  ),
  (
    title: 'Low Tide',
    artist: 'The Driftwood Trio',
    art: Icons.sailing,
    length: 187,
  ),
];

/// The suggestions in the expanded search bar, and the demo replies.
const _suggestions = [
  'Why is sea glass frosted?',
  'Rarest sea glass colors',
  'Where to find sea glass',
  'Which apps inspired this?',
];

const _replies = {
  'Why is sea glass frosted?':
      'Decades of waves and sand round off the edges, while the water slowly '
      'leaches soda and lime out of the surface. The tiny pits left behind '
      'scatter light, so the glass glows instead of shining.',
  'Rarest sea glass colors':
      'Red, orange and yellow are the rarest: few bottles were ever made in '
      'them. Cobalt and cornflower blue come next, and soft lavender is '
      'old glass the sun has turned.',
  'Where to find sea glass':
      'Look on pebbly beaches near old ports and towns, after a winter storm '
      'and at low tide, where the waves leave the heavier stones.',
  'Which apps inspired this?':
      'Material You: its tonal colors, shapes and motion, rendered here as '
      'tumbled glass with liquid_glass_renderer.',
};

const _menuItems = [
  (Icons.mic_none, 'Audio'),
  (Icons.image_outlined, 'Image'),
  (Icons.brush_outlined, 'Drawing'),
  (Icons.checklist, 'List'),
  (Icons.text_fields, 'Text'),
];

/// Where each component sits, in a canvas of [size] logical pixels, and
/// the scale [u] of its content.
enum _Layout {
  /// The landscape layout of the concept, in its own pixels.
  wide(
    size: Size(1300, 860),
    u: 1,
    rects: {
      'search': Rect.fromLTWH(44, 40, 1090, 92),
      'avatar': Rect.fromLTWH(1156, 41, 90, 90),
      'internet': Rect.fromLTWH(44, 200, 295, 104),
      'moon': Rect.fromLTWH(360, 199, 106, 106),
      'flashlight': Rect.fromLTWH(487, 199, 106, 106),
      'slider': Rect.fromLTWH(40, 354, 553, 82),
      'media': Rect.fromLTWH(44, 486, 547, 283),
      'day': Rect.fromLTWH(694, 198, 160, 78),
      'week': Rect.fromLTWH(862, 198, 216, 78),
      'month': Rect.fromLTWH(1086, 198, 160, 78),
      'toolbar': Rect.fromLTWH(697, 336, 320, 90),
      'newChat': Rect.fromLTWH(1039, 336, 207, 90),
      'nav': Rect.fromLTWH(689, 484, 559, 122),
      'fab': Rect.fromLTWH(1139, 664, 106, 106),
      'caption': Rect.fromLTWH(0, 800, 1300, 40),
    },
  ),

  /// One column for portrait phones.
  narrow(
    size: Size(440, 1010),
    u: 0.62,
    rects: {
      'search': Rect.fromLTWH(12, 24, 340, 62),
      'avatar': Rect.fromLTWH(366, 24, 62, 62),
      'internet': Rect.fromLTWH(12, 112, 214, 76),
      'moon': Rect.fromLTWH(240, 114, 72, 72),
      'flashlight': Rect.fromLTWH(326, 114, 72, 72),
      'slider': Rect.fromLTWH(10, 214, 418, 58),
      'day': Rect.fromLTWH(12, 300, 118, 58),
      'week': Rect.fromLTWH(136, 300, 168, 58),
      'month': Rect.fromLTWH(310, 300, 118, 58),
      'toolbar': Rect.fromLTWH(12, 386, 254, 66),
      'newChat': Rect.fromLTWH(278, 386, 150, 66),
      'media': Rect.fromLTWH(12, 480, 416, 226),
      'nav': Rect.fromLTWH(12, 734, 416, 92),
      'fab': Rect.fromLTWH(354, 852, 74, 74),
      'caption': Rect.fromLTWH(0, 950, 440, 50),
    },
  );

  const _Layout({required this.size, required this.u, required this.rects});

  final Size size;
  final double u;
  final Map<String, Rect> rects;

  Rect operator [](String id) => rects[id]!;
}

@visibleForTesting
class GoogleSeaGlassPageState extends State<GoogleSeaGlassPage>
    with TickerProviderStateMixin {
  /// The light rises over the first second and a half: long shadows swing
  /// in under the glass, as when the concept loads.
  late final AnimationController _intro = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1500),
  )..forward();

  /// The wavy progress line of the playing track.
  late final AnimationController _wave = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  );

  Offset? _pointer;
  bool _dusk = false;
  bool _flashlight = false;

  /// The slider rests where the concept's does; it brightens no further.
  double _brightness = 0.78;
  int _glass = 0;
  int? _undoGlass;

  bool _playing = false;
  int _track = 0;
  double _position = 37;
  Timer? _clock;

  int _segment = 1;
  int _tool = 0;
  int _tab = 0;

  bool _menuOpen = false;
  bool _searchOpen = false;
  bool _aiMode = false;
  String? _question;
  int _typedWords = 0;
  Timer? _typing;

  String? _toast;
  String _toastAction = 'OK';
  Timer? _toastTimer;

  /// The glass set in use.
  @visibleForTesting
  GoogleSeaGlass get glass => GoogleSeaGlass.values[_glass];

  @override
  void dispose() {
    _intro.dispose();
    _wave.dispose();
    _clock?.cancel();
    _typing?.cancel();
    _toastTimer?.cancel();
    super.dispose();
  }

  // MARK: Light

  /// The settings under the current light.
  @visibleForTesting
  LiquidGlassSettings settingsFor(Size screen) {
    final rise = Curves.easeOutCubic.transform(_intro.value);
    final finish = const SeaGlassFinish.google().copyWith(
      elevation: lerpDouble(80, 22, rise),
    );
    final level = 0.72 + 0.28 * (_brightness / 0.78).clamp(0.0, 1.0);
    final center = screen.center(Offset.zero);
    final pointer = _pointer ?? center;
    if (_flashlight) {
      return LiquidGlassSettings.googleSeaGlass(
        finish: finish,
        light: GlassLight(
          position: pointer,
          height: 360,
          reach: 560,
          color: const Color(0xFFFFF0DC),
          intensity: 0.9 * level,
          ambient: Color.from(
            alpha: 1,
            red: 0.44 * level,
            green: 0.37 * level,
            blue: 0.3 * level,
          ),
        ),
      );
    }
    // The pointer turns the daylight a little, as if it were moving the sun.
    final shift = (pointer.dx - center.dx) / math.max(screen.width, 1);
    return LiquidGlassSettings.googleSeaGlass(
      finish: finish,
      lightAngle: 0.6 * shift,
      light: GlassLight(
        color: _dusk ? const Color(0xFFFFE2C4) : const Color(0xFFFFFFFF),
        intensity: (_dusk ? 0.86 : 1) * level,
        ambient: _dusk ? const Color(0xFF1C120A) : const Color(0xFF000000),
      ),
    );
  }

  Color get _background {
    final level = 0.8 + 0.2 * (_brightness / 0.78).clamp(0.0, 1.0);
    final base = _flashlight
        ? const Color(0xFF6F5F51)
        : _dusk
        ? const Color(0xFFE6D0B6)
        : const Color(0xFFEFEFEC);
    return Color.lerp(const Color(0xFF000000), base, level)!;
  }

  // MARK: Actions

  void _showToast(String text, {String action = 'OK'}) {
    _toastTimer?.cancel();
    setState(() {
      _toast = text;
      _toastAction = action;
    });
    _toastTimer = Timer(const Duration(seconds: 3), () {
      if (mounted) setState(() => _toast = null);
    });
  }

  void _nextGlass() {
    setState(() {
      _undoGlass = _glass;
      _glass = (_glass + 1) % GoogleSeaGlass.values.length;
    });
    _showToast(glass.name, action: 'Undo');
  }

  void _toastPressed() {
    if (_toastAction == 'Undo' && _undoGlass != null) {
      setState(() => _glass = _undoGlass!);
    }
    _toastTimer?.cancel();
    setState(() => _toast = null);
  }

  void _togglePlay() {
    setState(() => _playing = !_playing);
    if (_playing) {
      unawaited(_wave.repeat());
      _clock = Timer.periodic(const Duration(seconds: 1), (_) {
        if (!mounted) return;
        setState(() {
          _position += 1;
          if (_position >= _tracks[_track].length) _skip(1);
        });
      });
    } else {
      _wave.stop();
      _clock?.cancel();
    }
  }

  void _skip(int direction) => setState(() {
    _track = (_track + direction) % _tracks.length;
    _position = 0;
  });

  void _openSearch() {
    if (_searchOpen) return;
    setState(() {
      _searchOpen = true;
      _menuOpen = false;
    });
  }

  void _closeOverlays() {
    _typing?.cancel();
    setState(() {
      _searchOpen = false;
      _question = null;
      _menuOpen = false;
    });
  }

  void _ask(String question) {
    _typing?.cancel();
    setState(() {
      _aiMode = true;
      _question = question;
      _typedWords = 0;
    });
    final words = _replies[question]!.split(' ').length;
    _typing = Timer.periodic(const Duration(milliseconds: 70), (timer) {
      if (!mounted) return;
      setState(() => _typedWords++);
      if (_typedWords >= words) timer.cancel();
    });
  }

  void _pick(String item) {
    setState(() => _menuOpen = false);
    _showToast('New ${item.toLowerCase()} · demo');
  }

  // MARK: Build

  @override
  Widget build(BuildContext context) {
    final screen = MediaQuery.sizeOf(context);
    final layout = screen.width >= screen.height
        ? _Layout.wide
        : _Layout.narrow;
    return MouseRegion(
      onHover: (event) => setState(() => _pointer = event.position),
      child: Listener(
        onPointerDown: (event) => setState(() => _pointer = event.position),
        onPointerMove: (event) => setState(() => _pointer = event.position),
        child: AnimatedBuilder(
          animation: _intro,
          builder: (context, _) => _scene(context, screen, layout),
        ),
      ),
    );
  }

  Widget _scene(BuildContext context, Size screen, _Layout layout) {
    final settings = settingsFor(screen);
    final pointer = _pointer ?? screen.center(Offset.zero);
    final spot = Alignment(
      pointer.dx / math.max(screen.width, 1) * 2 - 1,
      pointer.dy / math.max(screen.height, 1) * 2 - 1,
    );
    final background = _background;
    return Material(
      type: MaterialType.transparency,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 600),
        decoration: BoxDecoration(
          color: background,
          gradient: _flashlight
              ? RadialGradient(
                  center: spot,
                  radius: 0.75,
                  colors: [
                    Color.lerp(background, const Color(0xFFF6E7D2), 0.8)!,
                    background,
                    Color.lerp(background, const Color(0xFF000000), 0.35)!,
                  ],
                  stops: const [0, 0.55, 1],
                )
              : _dusk
              ? RadialGradient(
                  radius: 1.1,
                  colors: [
                    background,
                    Color.lerp(background, const Color(0xFF3A2414), 0.3)!,
                  ],
                )
              : null,
        ),
        child: Center(
          child: FittedBox(
            child: SizedBox.fromSize(
              size: layout.size,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  Positioned.fill(
                    child: GestureDetector(
                      behavior: HitTestBehavior.translucent,
                      onTap: _searchOpen || _menuOpen ? _closeOverlays : null,
                    ),
                  ),
                  Positioned.fill(
                    child: LiquidGlassLayer(
                      settings: settings,
                      child: _components(layout),
                    ),
                  ),
                  Positioned.fill(
                    child: LiquidGlassLayer(
                      settings: settings,
                      child: _floating(layout),
                    ),
                  ),
                  Positioned.fromRect(
                    rect: layout['caption'],
                    child: _Caption(
                      flashlight: _flashlight,
                      dark: _flashlight || _brightness < 0.4,
                      u: layout.u,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// The components in the main layer; they fade back while the search or
  /// the menu is open.
  Widget _components(_Layout layout) {
    final u = layout.u;
    final g = glass;
    final dimmed = _searchOpen || _menuOpen;

    final search = layout['search'];
    final searchRect = _searchOpen
        ? Rect.fromLTWH(
            search.left,
            search.top,
            search.width,
            search.height * (_question == null ? 3.4 : 3.2),
          )
        : search;
    double faded(String id) {
      if (!dimmed || (id == 'search' && _searchOpen)) return 1;
      // What the open search card covers fades out completely, so neither
      // its glass nor its content shows through the card.
      if (_searchOpen && layout[id].overlaps(searchRect)) return 0;
      return 0.15;
    }

    Widget glassAt(
      String id,
      LiquidShape shape,
      Color color,
      Widget child, {
      Rect? rect,
      VoidCallback? onTap,
    }) => _Glass(
      key: ValueKey(id),
      rect: rect ?? layout[id],
      shape: shape,
      color: color,
      visibility: faded(id),
      onTap: onTap,
      child: child,
    );

    final segments = [('day', 'Day'), ('week', 'Week'), ('month', 'Month')];
    return Stack(
      clipBehavior: Clip.none,
      children: [
        glassAt(
          'avatar',
          const LiquidOval(),
          g.avatar,
          Text(
            'G',
            style: TextStyle(
              color: const Color(0xFF8A1F1F),
              fontSize: 46 * u,
              fontWeight: FontWeight.w700,
            ),
          ),
          onTap: _nextGlass,
        ),
        glassAt(
          'internet',
          LiquidRoundedSuperellipse(borderRadius: 30 * u),
          g.primary,
          _Tile(
            u: u,
            icon: Icons.wifi,
            title: 'Internet',
            subtitle: 'Seaside Wi-Fi',
          ),
        ),
        glassAt(
          'moon',
          const LiquidOval(),
          _dusk ? g.accent : g.clear,
          Icon(
            _dusk ? Icons.dark_mode : Icons.dark_mode_outlined,
            size: 40 * u,
            color: _dusk ? const Color(0xFF5A3A00) : const Color(0xFF333B3F),
          ),
          onTap: () => setState(() => _dusk = !_dusk),
        ),
        glassAt(
          'flashlight',
          const LiquidOval(),
          _flashlight ? g.error : g.clear,
          Icon(
            _flashlight ? Icons.flashlight_on : Icons.flashlight_off,
            size: 40 * u,
            color: _flashlight
                ? const Color(0xFF6B1414)
                : const Color(0xFF333B3F),
          ),
          onTap: () => setState(() {
            _flashlight = !_flashlight;
            if (_flashlight) _dusk = true;
          }),
        ),
        glassAt(
          'slider',
          LiquidRoundedSuperellipse(borderRadius: 41 * u),
          g.tertiaryLight,
          _Slider(
            u: u,
            value: _brightness,
            fill: g.tertiary,
            onChanged: (value) => setState(() => _brightness = value),
          ),
        ),
        glassAt(
          'media',
          LiquidRoundedSuperellipse(borderRadius: 40 * u),
          g.secondary,
          _Media(
            u: u,
            track: _tracks[_track],
            position: _position,
            playing: _playing,
            wave: _wave,
            art: _track == 0 ? g.accent : g.tertiaryLight,
            chip: g.clear,
            onPlay: _togglePlay,
            onSkip: _skip,
          ),
        ),
        for (final (index, (id, label)) in segments.indexed)
          glassAt(
            id,
            index == 1
                ? LiquidRoundedSuperellipse(borderRadius: 39 * u)
                : LiquidRoundedSuperellipse(borderRadius: 20 * u),
            _segment == index ? g.positive : g.positiveLight,
            _Segment(u: u, label: label, selected: _segment == index),
            onTap: () => setState(() => _segment = index),
          ),
        glassAt(
          'toolbar',
          LiquidRoundedSuperellipse(borderRadius: 45 * u),
          g.secondary,
          _Toolbar(
            u: u,
            selected: _tool,
            chip: g.clear,
            onSelect: (index) => setState(() => _tool = index),
          ),
        ),
        glassAt(
          'newChat',
          LiquidRoundedSuperellipse(borderRadius: 40 * u),
          g.error,
          _Label(
            u: u,
            icon: Icons.edit,
            text: 'New chat',
            color: const Color(0xFF6B1414),
          ),
          onTap: () => _showToast('New chat · demo'),
        ),
        glassAt(
          'nav',
          LiquidRoundedSuperellipse(borderRadius: 54 * u),
          g.tertiary,
          _NavBar(
            u: u,
            selected: _tab,
            chip: g.clear,
            onSelect: (index) => setState(() => _tab = index),
          ),
        ),
        // Last, so the open search card and its content cover the faded
        // components it grows over.
        glassAt(
          'search',
          LiquidRoundedSuperellipse(borderRadius: 46 * u),
          g.primary,
          _SearchContent(
            u: u,
            open: _searchOpen,
            aiMode: _aiMode,
            question: _question,
            typedWords: _typedWords,
            onOpen: _openSearch,
            onAiMode: () => setState(() => _aiMode = !_aiMode),
            onAsk: _ask,
            chip: g.clear,
          ),
          rect: searchRect,
        ),
      ],
    );
  }

  /// The floating action button, its menu and the toast, in the second
  /// layer.
  Widget _floating(_Layout layout) {
    final u = layout.u;
    final g = glass;
    final fab = layout['fab'];
    final itemHeight = 62 * u;
    final caption = layout['caption'];
    return Stack(
      clipBehavior: Clip.none,
      children: [
        for (final (index, (icon, label)) in _menuItems.indexed)
          _MenuItem(
            key: ValueKey(label),
            open: _menuOpen,
            // Staggered from the button upward.
            delay: (_menuItems.length - 1 - index) * 0.08,
            rect: Rect.fromLTWH(
              fab.right - 190 * u,
              fab.top - (_menuItems.length - index) * (itemHeight + 10 * u),
              190 * u,
              itemHeight,
            ),
            from: fab,
            color: g.accent,
            u: u,
            icon: icon,
            label: label,
            onTap: () => _pick(label),
          ),
        _Glass(
          key: const ValueKey('fab'),
          rect: fab,
          shape: LiquidRoundedSuperellipse(borderRadius: 34 * u),
          color: g.accent,
          onTap: () => setState(() {
            _menuOpen = !_menuOpen;
            _searchOpen = false;
          }),
          child: AnimatedRotation(
            turns: _menuOpen ? 0.125 : 0,
            duration: const Duration(milliseconds: 250),
            child: Icon(
              Icons.add,
              size: 44 * u,
              color: const Color(0xFF4A3000),
            ),
          ),
        ),
        if (_toast case final toast?)
          _Glass(
            key: const ValueKey('toast'),
            rect: Rect.fromCenter(
              center: caption.center.translate(0, -60 * u),
              width: 330 * u,
              height: 64 * u,
            ),
            shape: LiquidRoundedSuperellipse(borderRadius: 32 * u),
            color: g.clear,
            child: _Toast(
              u: u,
              text: toast,
              action: _toastAction,
              onAction: _toastPressed,
            ),
          ),
      ],
    );
  }
}

// MARK: Components

/// One component: a sea glass shape whose rect, color and visibility
/// animate.
class _Glass extends StatelessWidget {
  const _Glass({
    required this.rect,
    required this.shape,
    required this.color,
    required this.child,
    this.visibility = 1,
    this.onTap,
    super.key,
  });

  final Rect rect;
  final LiquidShape shape;
  final Color color;
  final double visibility;
  final VoidCallback? onTap;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return AnimatedPositioned.fromRect(
      rect: rect,
      duration: const Duration(milliseconds: 320),
      curve: Curves.easeOutBack,
      child: TweenAnimationBuilder<Color?>(
        tween: ColorTween(end: color),
        duration: const Duration(milliseconds: 450),
        builder: (context, color, _) => TweenAnimationBuilder<double>(
          tween: Tween(end: visibility),
          duration: const Duration(milliseconds: 260),
          builder: (context, visibility, _) => LiquidGlass(
            shape: shape,
            appearance: LiquidGlassAppearance.seaGlass(
              color: color!,
              visibility: visibility,
            ),
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: onTap,
              child: Center(child: child),
            ),
          ),
        ),
      ),
    );
  }
}

/// A milky glass chip on another glass: the selection on toolbars, the
/// play button, the AI Mode button.
class _Chip extends StatelessWidget {
  const _Chip({
    required this.color,
    required this.child,
    required this.radius,
    this.padding = EdgeInsets.zero,
    this.milk = 0.6,
  });

  final Color color;
  final Widget child;
  final double radius;
  final EdgeInsets padding;

  /// How far the chip's color is mixed toward white.
  final double milk;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Color.lerp(color, const Color(0xFFFFFFFF), milk)!.withValues(
          alpha: 0.92,
        ),
        borderRadius: BorderRadius.circular(radius),
        boxShadow: const [
          BoxShadow(
            color: Color(0x22000000),
            blurRadius: 10,
            offset: Offset(2, 3),
          ),
        ],
      ),
      child: Padding(padding: padding, child: child),
    );
  }
}

const _ink = Color(0xFF1F2A6B);

class _SearchContent extends StatelessWidget {
  const _SearchContent({
    required this.u,
    required this.open,
    required this.aiMode,
    required this.question,
    required this.typedWords,
    required this.onOpen,
    required this.onAiMode,
    required this.onAsk,
    required this.chip,
  });

  final double u;
  final bool open;
  final bool aiMode;
  final String? question;
  final int typedWords;
  final VoidCallback onOpen;
  final VoidCallback onAiMode;
  final ValueChanged<String> onAsk;
  final Color chip;

  @override
  Widget build(BuildContext context) {
    final bar = SizedBox(
      height: 92 * u,
      child: Row(
        children: [
          SizedBox(width: 26 * u),
          Icon(Icons.search, size: 34 * u, color: _ink),
          SizedBox(width: 16 * u),
          Expanded(
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: onOpen,
              child: Text(
                aiMode ? 'Ask AI Mode' : 'Search',
                maxLines: 1,
                style: TextStyle(
                  color: _ink.withValues(alpha: 0.75),
                  fontSize: 28 * u,
                ),
              ),
            ),
          ),
          Icon(Icons.mic, size: 32 * u, color: _ink),
          SizedBox(width: 30 * u),
          Icon(Icons.photo_camera_outlined, size: 32 * u, color: _ink),
          SizedBox(width: 20 * u),
          GestureDetector(
            onTap: onAiMode,
            child: _Chip(
              color: aiMode ? const Color(0xFFF2C8F4) : chip,
              radius: 34 * u,
              padding: EdgeInsets.symmetric(
                horizontal: 20 * u,
                vertical: 14 * u,
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.auto_awesome, size: 26 * u, color: _ink),
                  SizedBox(width: 8 * u),
                  Text(
                    'AI Mode',
                    style: TextStyle(
                      color: _ink,
                      fontSize: 24 * u,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),
          SizedBox(width: 12 * u),
        ],
      ),
    );
    if (!open) return bar;
    final reply = question == null ? null : _replies[question]!.split(' ');
    // The glass grows and shrinks around the content, which stays put.
    return ClipRect(
      child: OverflowBox(
        alignment: Alignment.topLeft,
        maxHeight: double.infinity,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            bar,
            Padding(
              padding: EdgeInsets.fromLTRB(30 * u, 4 * u, 30 * u, 0),
              child: reply == null
                  ? Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'TRY SEARCHING',
                          style: TextStyle(
                            color: _ink.withValues(alpha: 0.6),
                            fontSize: 15 * u,
                            letterSpacing: 1.2,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        SizedBox(height: 12 * u),
                        Wrap(
                          spacing: 12 * u,
                          runSpacing: 12 * u,
                          children: [
                            for (final suggestion in _suggestions)
                              GestureDetector(
                                onTap: () => onAsk(suggestion),
                                child: _Chip(
                                  color: chip,
                                  radius: 24 * u,
                                  padding: EdgeInsets.symmetric(
                                    horizontal: 18 * u,
                                    vertical: 10 * u,
                                  ),
                                  child: Text(
                                    suggestion,
                                    style: TextStyle(
                                      color: _ink,
                                      fontSize: 20 * u,
                                    ),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ],
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'AI MODE · DEMO REPLY',
                          style: TextStyle(
                            color: _ink.withValues(alpha: 0.6),
                            fontSize: 15 * u,
                            letterSpacing: 1.2,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        SizedBox(height: 10 * u),
                        Text(
                          question!,
                          style: TextStyle(
                            color: _ink,
                            fontSize: 22 * u,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        SizedBox(height: 8 * u),
                        Text(
                          reply.take(typedWords).join(' '),
                          style: TextStyle(
                            color: _ink,
                            fontSize: 20 * u,
                            height: 1.35,
                          ),
                        ),
                      ],
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Tile extends StatelessWidget {
  const _Tile({
    required this.u,
    required this.icon,
    required this.title,
    required this.subtitle,
  });

  final double u;
  final IconData icon;
  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(width: 28 * u),
        Icon(icon, size: 40 * u, color: _ink),
        SizedBox(width: 22 * u),
        Flexible(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: _ink,
                  fontSize: 24 * u,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                subtitle,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: _ink.withValues(alpha: 0.8),
                  fontSize: 19 * u,
                ),
              ),
            ],
          ),
        ),
        SizedBox(width: 16 * u),
      ],
    );
  }
}

class _Slider extends StatelessWidget {
  const _Slider({
    required this.u,
    required this.value,
    required this.fill,
    required this.onChanged,
  });

  final double u;
  final double value;
  final Color fill;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final start = 18 * u;
        final track = width - 2 * start;
        void update(Offset local) =>
            onChanged(((local.dx - start) / track).clamp(0.0, 1.0));
        final x = start + track * value;
        return GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTapDown: (details) => update(details.localPosition),
          onHorizontalDragUpdate: (details) => update(details.localPosition),
          child: Stack(
            children: [
              Positioned(
                left: start * 0.5,
                top: constraints.maxHeight * 0.16,
                bottom: constraints.maxHeight * 0.16,
                width: math.max(x - start * 0.5, 0),
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: fill.withValues(alpha: 0.55),
                    borderRadius: BorderRadius.horizontal(
                      left: Radius.circular(30 * u),
                    ),
                  ),
                ),
              ),
              Positioned(
                left: x - 3 * u,
                top: -4 * u,
                bottom: -4 * u,
                width: 6 * u,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: Color.lerp(fill, const Color(0xFF0F5652), 0.45),
                    borderRadius: BorderRadius.circular(3 * u),
                  ),
                ),
              ),
              Positioned(
                left: 24 * u,
                top: 0,
                bottom: 0,
                child: Icon(
                  Icons.light_mode,
                  size: 32 * u,
                  color: const Color(0xFF0F4F4C),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _Media extends StatelessWidget {
  const _Media({
    required this.u,
    required this.track,
    required this.position,
    required this.playing,
    required this.wave,
    required this.art,
    required this.chip,
    required this.onPlay,
    required this.onSkip,
  });

  final double u;
  final ({String title, String artist, IconData art, int length}) track;
  final double position;
  final bool playing;
  final Animation<double> wave;
  final Color art;
  final Color chip;
  final VoidCallback onPlay;
  final ValueChanged<int> onSkip;

  static const _purple = Color(0xFF5B1670);

  String _time(double seconds) =>
      '${seconds ~/ 60}:${(seconds % 60).floor().toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    final progress = (position / track.length).clamp(0.0, 1.0);
    return Padding(
      padding: EdgeInsets.fromLTRB(26 * u, 24 * u, 26 * u, 22 * u),
      child: Column(
        children: [
          Row(
            children: [
              _Chip(
                color: art,
                milk: 0.05,
                radius: 18 * u,
                child: SizedBox.square(
                  dimension: 76 * u,
                  child: Icon(
                    track.art,
                    size: 40 * u,
                    color: Color.lerp(art, const Color(0xFF3A2405), 0.75),
                  ),
                ),
              ),
              SizedBox(width: 22 * u),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      track.title,
                      style: TextStyle(
                        color: _purple,
                        fontSize: 26 * u,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      track.artist,
                      style: TextStyle(
                        color: _purple.withValues(alpha: 0.8),
                        fontSize: 21 * u,
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.headphones_outlined, size: 32 * u, color: _purple),
            ],
          ),
          const Spacer(),
          Row(
            children: [
              Text(
                _time(position),
                style: TextStyle(color: _purple, fontSize: 19 * u),
              ),
              SizedBox(width: 18 * u),
              Expanded(
                child: SizedBox(
                  height: 24 * u,
                  child: AnimatedBuilder(
                    animation: wave,
                    builder: (context, _) => CustomPaint(
                      painter: _ProgressPainter(
                        progress: progress,
                        phase: wave.value,
                        wavy: playing,
                        color: _purple,
                        u: u,
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(width: 18 * u),
              Text(
                _time(track.length.toDouble()),
                style: TextStyle(color: _purple, fontSize: 19 * u),
              ),
            ],
          ),
          const Spacer(),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              GestureDetector(
                onTap: () => onSkip(-1),
                child: Icon(Icons.skip_previous, size: 40 * u, color: _purple),
              ),
              SizedBox(width: 70 * u),
              GestureDetector(
                onTap: onPlay,
                child: _Chip(
                  color: chip,
                  radius: playing ? 22 * u : 48 * u,
                  child: SizedBox.square(
                    dimension: 96 * u,
                    child: Icon(
                      playing ? Icons.pause : Icons.play_arrow,
                      size: 44 * u,
                      color: _purple,
                    ),
                  ),
                ),
              ),
              SizedBox(width: 70 * u),
              GestureDetector(
                onTap: () => onSkip(1),
                child: Icon(Icons.skip_next, size: 40 * u, color: _purple),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// A progress line that turns into a moving wave while the track plays.
class _ProgressPainter extends CustomPainter {
  const _ProgressPainter({
    required this.progress,
    required this.phase,
    required this.wavy,
    required this.color,
    required this.u,
  });

  final double progress;
  final double phase;
  final bool wavy;
  final Color color;
  final double u;

  @override
  void paint(Canvas canvas, Size size) {
    final y = size.height / 2;
    final played = size.width * progress;
    final stroke = Paint()
      ..color = color
      ..strokeWidth = 4 * u
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    final path = Path()..moveTo(0, y);
    final amplitude = wavy ? 5 * u : 0.0;
    for (var x = 0.0; x <= played; x += 2) {
      path.lineTo(
        x,
        y + amplitude * math.sin((x / (26 * u) - phase) * 2 * math.pi),
      );
    }
    canvas
      ..drawPath(path, stroke)
      ..drawLine(
        Offset(played + 8 * u, y),
        Offset(size.width, y),
        stroke..color = color.withValues(alpha: 0.35),
      )
      ..drawLine(
        Offset(played, y - 11 * u),
        Offset(played, y + 11 * u),
        Paint()
          ..color = color
          ..strokeWidth = 6 * u
          ..strokeCap = StrokeCap.round,
      );
  }

  @override
  bool shouldRepaint(_ProgressPainter oldDelegate) =>
      oldDelegate.progress != progress ||
      oldDelegate.phase != phase ||
      oldDelegate.wavy != wavy ||
      oldDelegate.color != color;
}

class _Segment extends StatelessWidget {
  const _Segment({required this.u, required this.label, this.selected = false});

  final double u;
  final String label;
  final bool selected;

  static const _green = Color(0xFF1F5A2A);

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (selected) ...[
          Icon(Icons.check, size: 26 * u, color: _green),
          SizedBox(width: 8 * u),
        ],
        Text(
          label,
          style: TextStyle(
            color: _green,
            fontSize: 24 * u,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}

class _Toolbar extends StatelessWidget {
  const _Toolbar({
    required this.u,
    required this.selected,
    required this.chip,
    required this.onSelect,
  });

  final double u;
  final int selected;
  final Color chip;
  final ValueChanged<int> onSelect;

  static const _icons = [
    Icons.home,
    Icons.chat_bubble_outline,
    Icons.groups,
    Icons.more_horiz,
  ];

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final slot = (constraints.maxWidth - 20 * u) / _icons.length;
        return Stack(
          children: [
            AnimatedPositioned(
              duration: const Duration(milliseconds: 320),
              curve: Curves.easeOutBack,
              left: 10 * u + slot * selected + (slot - 72 * u) / 2,
              top: (constraints.maxHeight - 72 * u) / 2,
              child: _Chip(
                color: chip,
                radius: 36 * u,
                child: SizedBox.square(dimension: 72 * u),
              ),
            ),
            Row(
              children: [
                SizedBox(width: 10 * u),
                for (final (index, icon) in _icons.indexed)
                  SizedBox(
                    width: slot,
                    height: constraints.maxHeight,
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onTap: () => onSelect(index),
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          Icon(icon, size: 34 * u, color: _Media._purple),
                          if (index == 1)
                            Positioned(
                              top: constraints.maxHeight * 0.2,
                              right: slot * 0.24,
                              child: Container(
                                width: 12 * u,
                                height: 12 * u,
                                decoration: const BoxDecoration(
                                  color: Color(0xFFD93025),
                                  shape: BoxShape.circle,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ],
        );
      },
    );
  }
}

class _Label extends StatelessWidget {
  const _Label({
    required this.u,
    required this.icon,
    required this.text,
    required this.color,
  });

  final double u;
  final IconData icon;
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    // Shrinks rather than overflows with larger text.
    return FittedBox(
      fit: BoxFit.scaleDown,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 30 * u, color: color),
          SizedBox(width: 14 * u),
          Text(
            text,
            style: TextStyle(
              color: color,
              fontSize: 24 * u,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _NavBar extends StatelessWidget {
  const _NavBar({
    required this.u,
    required this.selected,
    required this.chip,
    required this.onSelect,
  });

  final double u;
  final int selected;
  final Color chip;
  final ValueChanged<int> onSelect;

  static const _teal = Color(0xFF0F4F4C);
  static const _items = [
    (Icons.home, 'Home'),
    (Icons.dialpad, 'Keypad'),
    (Icons.voicemail, 'Voicemail'),
  ];

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final slot = (constraints.maxWidth - 40 * u) / _items.length;
        return Stack(
          children: [
            AnimatedPositioned(
              duration: const Duration(milliseconds: 320),
              curve: Curves.easeOutBack,
              left: 20 * u + slot * selected + (slot - 80 * u) / 2,
              top: 22 * u,
              child: _Chip(
                color: chip,
                radius: 22 * u,
                child: SizedBox(width: 80 * u, height: 44 * u),
              ),
            ),
            Row(
              children: [
                SizedBox(width: 20 * u),
                for (final (index, (icon, label)) in _items.indexed)
                  SizedBox(
                    width: slot,
                    child: GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onTap: () => onSelect(index),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            height: 44 * u,
                            child: Badge(
                              isLabelVisible: index == 2,
                              label: const Text('2'),
                              backgroundColor: _teal,
                              child: Icon(icon, size: 30 * u, color: _teal),
                            ),
                          ),
                          SizedBox(height: 8 * u),
                          Text(
                            label,
                            style: TextStyle(
                              color: _teal,
                              fontSize: 20 * u,
                              fontWeight: index == selected
                                  ? FontWeight.w700
                                  : FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ],
        );
      },
    );
  }
}

/// A menu item that springs out of the floating action button.
class _MenuItem extends StatelessWidget {
  const _MenuItem({
    required this.open,
    required this.delay,
    required this.rect,
    required this.from,
    required this.color,
    required this.u,
    required this.icon,
    required this.label,
    required this.onTap,
    super.key,
  });

  final bool open;
  final double delay;
  final Rect rect;
  final Rect from;
  final Color color;
  final double u;
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(end: open ? 1 : 0),
      duration: Duration(milliseconds: 420 + (delay * 1000).round()),
      curve: Interval(open ? delay : 0, 1, curve: Curves.easeOutBack),
      builder: (context, t, _) {
        if (t <= 0.001) return const SizedBox.shrink();
        final start = Rect.fromCenter(
          center: from.center,
          width: from.width * 0.6,
          height: from.height * 0.6,
        );
        return Positioned.fromRect(
          rect: Rect.lerp(start, rect, t)!,
          child: LiquidGlass(
            shape: LiquidRoundedSuperellipse(borderRadius: 31 * u),
            appearance: LiquidGlassAppearance.seaGlass(
              color: color,
              visibility: t.clamp(0.0, 1.0),
            ),
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: onTap,
              child: Center(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  child: _Label(
                    u: u,
                    icon: icon,
                    text: label,
                    color: const Color(0xFF4A3000),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _Toast extends StatelessWidget {
  const _Toast({
    required this.u,
    required this.text,
    required this.action,
    required this.onAction,
  });

  final double u;
  final String text;
  final String action;
  final VoidCallback onAction;

  @override
  Widget build(BuildContext context) {
    // Shrinks rather than overflows with larger text.
    return FittedBox(
      fit: BoxFit.scaleDown,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            text,
            style: TextStyle(color: const Color(0xFF333B3F), fontSize: 20 * u),
          ),
          SizedBox(width: 20 * u),
          GestureDetector(
            onTap: onAction,
            child: Text(
              action,
              style: TextStyle(
                color: const Color(0xFF1F2A2E),
                fontSize: 20 * u,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Caption extends StatelessWidget {
  const _Caption({
    required this.flashlight,
    required this.dark,
    required this.u,
  });

  final bool flashlight;
  final bool dark;
  final double u;

  @override
  Widget build(BuildContext context) {
    final color = dark ? const Color(0xCCFFFFFF) : const Color(0xFF6B6B6B);
    final style = TextStyle(color: color, fontSize: 20 * math.max(u, 0.7));
    return Center(
      child: Text.rich(
        TextSpan(
          style: style,
          children: [
            const TextSpan(text: 'Unofficial Material concept · '),
            TextSpan(
              text: flashlight
                  ? 'Move your pointer to aim the flashlight'
                  : 'Move your pointer to shift the light',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            const TextSpan(text: ' · Tap G for other glass'),
          ],
        ),
        textAlign: TextAlign.center,
      ),
    );
  }
}
