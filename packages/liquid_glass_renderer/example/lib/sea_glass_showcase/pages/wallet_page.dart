import 'package:flutter/material.dart' show Icons;
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/pages/common.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_options.dart';
import 'package:liquid_glass_renderer_example/sea_glass_showcase/showcase_style.dart';

/// A wallet of cards cut from sea glass, each a different color; the first
/// follows the palette.
class WalletPage extends StatefulWidget {
  const WalletPage({super.key});

  @override
  State<WalletPage> createState() => _WalletPageState();
}

class _WalletPageState extends State<WalletPage>
    with AutomaticKeepAliveClientMixin {
  final _cards = PageController(viewportFraction: .86);
  var _card = 0;

  static const _fixedCards = [
    ('Travel', SeaGlassColors.cobalt, '7310', r'$3,204.10'),
    ('Rewards', SeaGlassColors.amber, '0952', r'$1,876.44'),
    ('Savings', SeaGlassColors.seafoam, '6648', r'$5,121.00'),
  ];

  static const _activity = [
    (Icons.local_cafe_rounded, 'Driftwood Café', 'Today, 9:14', r'-$6.40'),
    (Icons.directions_boat_rounded, 'Harbour Ferry', 'Yesterday', r'-$12.00'),
    (Icons.payments_rounded, 'Salary', 'Oct 1', r'+$4,200.00'),
    (Icons.shopping_bag_rounded, 'Tide Market', 'Sep 29', r'-$48.75'),
  ];

  static const _actionColors = [
    SeaGlassColors.aqua,
    SeaGlassColors.lavender,
    SeaGlassColors.citron,
    SeaGlassColors.cornflower,
  ];

  @override
  bool get wantKeepAlive => true;

  @override
  void dispose() {
    _cards.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final style = ShowcaseScope.of(context);
    final ink = style.ink;
    final cards = [
      ('Everyday', style.accent, '4821', r'$2,278.78'),
      ..._fixedCards,
    ];
    return ShowcasePageFrame(
      title: 'Wallet',
      subtitle: 'Cards cut from sea glass',
      children: [
        ShowcaseLayer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              LayoutBuilder(
                builder: (context, constraints) {
                  final height = constraints.maxWidth * .86 / 1.586;
                  return SizedBox(
                    height: height + 20,
                    child: PageView.builder(
                      controller: _cards,
                      clipBehavior: Clip.none,
                      itemCount: cards.length,
                      onPageChanged: (index) => setState(() => _card = index),
                      itemBuilder: (context, index) => AnimatedBuilder(
                        animation: _cards,
                        builder: (context, child) {
                          final page =
                              _cards.hasClients &&
                                  _cards.position.haveDimensions
                              ? _cards.page ?? 0
                              : 0.0;
                          final distance = (page - index).abs().clamp(0, 1);
                          return Transform.scale(
                            scale: 1 - .08 * distance,
                            child: child,
                          );
                        },
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 10,
                          ),
                          child: _GlassCardFace(
                            name: cards[index].$1,
                            body: style.bodyOf(cards[index].$2),
                            last4: cards[index].$3,
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  for (var i = 0; i < cards.length; i++)
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 260),
                      width: i == _card ? 18 : 6,
                      height: 6,
                      margin: const EdgeInsets.symmetric(horizontal: 3),
                      decoration: BoxDecoration(
                        color: ink.withValues(alpha: i == _card ? .75 : .3),
                        borderRadius: const BorderRadius.all(
                          Radius.circular(3),
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 22),
              Center(
                child: Column(
                  children: [
                    Text(
                      '${cards[_card].$1} balance',
                      style: TextStyle(
                        color: ink.withValues(alpha: .7),
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 2),
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 250),
                      child: Text(
                        cards[_card].$4,
                        key: ValueKey(_card),
                        style: TextStyle(
                          color: ink,
                          fontSize: 40,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -1,
                          fontFeatures: const [FontFeature.tabularFigures()],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 22),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  for (final (i, (icon, label)) in const [
                    (Icons.north_east_rounded, 'Send'),
                    (Icons.south_west_rounded, 'Request'),
                    (Icons.add_rounded, 'Top up'),
                    (Icons.more_horiz_rounded, 'More'),
                  ].indexed)
                    Column(
                      children: [
                        GlassPebble(
                          icon: icon,
                          label: label,
                          size: 60,
                          body: style.bodyOf(_actionColors[i]),
                          onTap: () {},
                        ),
                        const SizedBox(height: 8),
                        Text(
                          label,
                          style: TextStyle(
                            color: ink,
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                ],
              ),
              const SizedBox(height: 28),
              const SectionHeader('Recent activity', trailing: 'See all'),
              GlassCard(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 6,
                ),
                child: Column(
                  children: [
                    for (final (icon, merchant, date, amount) in _activity)
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        child: Row(
                          children: [
                            IconBubble(icon: icon, size: 42),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    merchant,
                                    style: TextStyle(
                                      color: ink,
                                      fontSize: 15,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  Text(
                                    date,
                                    style: TextStyle(
                                      color: ink.withValues(alpha: .65),
                                      fontSize: 13,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            Text(
                              amount,
                              style: TextStyle(
                                color: ink,
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                                fontFeatures: const [
                                  FontFeature.tabularFigures(),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// The face of a payment card in sea glass of [body].
class _GlassCardFace extends StatelessWidget {
  const _GlassCardFace({
    required this.name,
    required this.body,
    required this.last4,
  });

  final String name;
  final Color body;
  final String last4;

  @override
  Widget build(BuildContext context) {
    final ink = inkOn(body);
    final small = TextStyle(
      color: ink.withValues(alpha: .8),
      fontSize: 12,
      fontWeight: FontWeight.w700,
      letterSpacing: 1.4,
    );
    return Semantics(
      label: '$name card ending in $last4',
      excludeSemantics: true,
      child: LiquidStretch.seaGlass(
        child: GlassCard(
          radius: 26,
          body: body,
          elevation: 9,
          padding: const EdgeInsets.fromLTRB(22, 20, 22, 18),
          child: DefaultTextStyle.merge(
            style: TextStyle(shadows: haloFor(ink)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text('SEA · GLASS', style: small),
                    const Spacer(),
                    Icon(Icons.contactless_rounded, color: ink, size: 26),
                  ],
                ),
                const SizedBox(height: 14),
                Container(
                  width: 42,
                  height: 32,
                  decoration: BoxDecoration(
                    borderRadius: const BorderRadius.all(Radius.circular(7)),
                    gradient: const LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [Color(0xFFF6E3A1), Color(0xFFC9A44C)],
                    ),
                    border: Border.all(
                      color: const Color(0x55000000),
                      width: .5,
                    ),
                  ),
                ),
                const Spacer(),
                FittedBox(
                  child: Text(
                    '••••   ••••   ••••   $last4',
                    style: TextStyle(
                      color: ink,
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 2,
                      fontFeatures: const [FontFeature.tabularFigures()],
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Text(name.toUpperCase(), style: small),
                    const Spacer(),
                    Text('09/29', style: small),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
