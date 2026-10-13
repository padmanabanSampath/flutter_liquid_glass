import 'dart:async';

import 'package:flutter/material.dart' show Icons;
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/common.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/quick_settings.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// The notification shade: a clock over quick settings tiles. The
/// brightness slider dims the wallpaper of every page.
class ShadePage extends StatefulWidget {
  const ShadePage({super.key});

  @override
  State<ShadePage> createState() => _ShadePageState();
}

class _ShadePageState extends State<ShadePage>
    with AutomaticKeepAliveClientMixin {
  final _active = <String>{'Wi-Fi', 'Location'};

  /// The slab glass role of each tile while it is on.
  static const _tileRoles = {
    'Wi-Fi': SlabRole.primary,
    'Mobile data': SlabRole.tertiary,
    'Bluetooth': SlabRole.secondary,
    'Torch': SlabRole.action,
    'Location': SlabRole.positive,
    'Modes': SlabRole.secondary,
    'Screen record': SlabRole.action,
    'Airplane mode': SlabRole.tertiary,
    'Scan QR code': SlabRole.positive,
    'Quick Share': SlabRole.primary,
  };

  @override
  bool get wantKeepAlive => true;

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final style = ShowcaseScope.of(context);
    return SafeArea(
      bottom: false,
      child: LayoutBuilder(
        builder: (context, viewport) => SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, tabBarClearance),
          child: ConstrainedBox(
            constraints: BoxConstraints(
              minHeight: (viewport.maxHeight - tabBarClearance - 16).clamp(
                0,
                double.infinity,
              ),
            ),
            child: Align(
              alignment: const Alignment(0, -.3),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 440),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const _Clock(),
                    const SizedBox(height: 32),
                    ShowcaseLayer(
                      child: QuickSettingsPanel(
                        body: style.roleBody(SlabRole.primary),
                        tileBody: (label) => style.roleBody(
                          _tileRoles[label] ?? SlabRole.tertiary,
                        ),
                        active: _active,
                        onToggle: (label) => setState(() {
                          if (!_active.remove(label)) _active.add(label);
                        }),
                        brightness: style.brightness,
                        onBrightness: (value) => ShowcaseScope.update(
                          context,
                          (style) => style.copyWith(brightness: value),
                        ),
                        darkWallpaper: style.dark,
                      ),
                    ),
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

/// A large clock and date over a small status row.
class _Clock extends StatefulWidget {
  const _Clock();

  @override
  State<_Clock> createState() => _ClockState();
}

class _ClockState extends State<_Clock> {
  late final Timer _tick;

  static const _weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  static const _months = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ];

  @override
  void initState() {
    super.initState();
    _tick = Timer.periodic(
      const Duration(seconds: 10),
      (_) => setState(() {}),
    );
  }

  @override
  void dispose() {
    _tick.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ink = ShowcaseScope.of(context).ink;
    final now = DateTime.now();
    final hour = now.hour % 12 == 0 ? 12 : now.hour % 12;
    final minute = now.minute.toString().padLeft(2, '0');
    final small = TextStyle(
      color: ink,
      fontSize: 14,
      fontWeight: FontWeight.w500,
    );
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$hour:$minute',
            style: TextStyle(
              color: ink,
              fontSize: 64,
              height: 1,
              fontWeight: FontWeight.w300,
              letterSpacing: -2,
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Text(
                '${_weekdays[now.weekday - 1]}, '
                '${_months[now.month - 1]} ${now.day}',
                style: small,
              ),
              const Spacer(),
              Icon(Icons.signal_cellular_alt_rounded, color: ink, size: 16),
              const SizedBox(width: 4),
              Icon(Icons.wifi_rounded, color: ink, size: 16),
              const SizedBox(width: 6),
              Text('82%', style: small),
            ],
          ),
        ],
      ),
    );
  }
}
