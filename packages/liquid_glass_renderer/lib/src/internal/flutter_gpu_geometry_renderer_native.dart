import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_gpu/gpu.dart' as gpu;

/// Renders the liquid glass geometry SDF shader using flutter_gpu.
///
/// Each output (matte, material map) cycles through a ring of textures that
/// only grow. A render draws into the top-left sub-rect of a texture no
/// in-flight frame reads and reports the texture size, so the final shader
/// scales its UVs to that sub-rect. Resizing geometry therefore reuses the
/// same textures instead of allocating, which matters because a
/// `gpu.Texture` is only freed when the GC finalizes its wrapper. The layer
/// reuses the image without calling [render] when geometry is unchanged
/// (including uniform translation).
///
/// This renderer owns the image handles returned by [gpu.Texture.asImage].
/// Replacing/disposal releases those handles; recorded scenes hold independent
/// native references. A scene rasterized more than [reuseAfterFrames] frames
/// after it was recorded may show a newer matte.
@internal
class FlutterGpuGeometryRenderer {
  FlutterGpuGeometryRenderer._fromShared(_SharedGeometryResources resources) {
    _pipeline = resources.pipeline;
    _materialGradientPipeline = resources.materialGradientPipeline;
    _materialTintGradientPipeline = resources.materialTintGradientPipeline;
    _copyPipeline = resources.copyPipeline;
    _copyUniformSlot = resources.copyUniformSlot;
    _copyTextureSlot = resources.copyTextureSlot;
    _copyUniformSize = resources.copyUniformSize;
    _uniformSlot = resources.uniformSlot;
    _uniformSize = resources.uniformSize;
    _offsetUOffset = resources.offsetUOffset;
    _offsetUTextureSize = resources.offsetUTextureSize;
    _offsetOpticalProps = resources.offsetOpticalProps;
    _offsetContourProps = resources.offsetContourProps;
    _offsetWobble = resources.offsetWobble;
    _offsetShapeData = resources.offsetShapeData;
    _offsetRseData = resources.offsetRseData;
    _offsetShapeTints = resources.offsetShapeTints;
    _offsetShapeResponses = resources.offsetShapeResponses;
    _offsetShapeBounds = resources.offsetShapeBounds;
    _vertexBufferView = resources.vertexBufferView;
    _uniformData = ByteData(_uniformSize);
    assert(() {
      _debugActiveRendererCount++;
      return true;
    }(), 'Track live geometry renderers in debug builds.');
  }

  // Flutter's centered half-pixel coverage transition.
  static const double _geometryAaHalfWidth = 0.5;

  static Future<FlutterGpuGeometryRenderer> fromAsset(String assetKey) async {
    final cachedResources = _resolvedAssetResources[assetKey];
    if (cachedResources != null) {
      try {
        return FlutterGpuGeometryRenderer._fromShared(cachedResources);
      } on Object {
        // Drop resources that failed to build a renderer so the next layer
        // reloads the bundle instead of failing the same way.
        if (identical(_resolvedAssetResources[assetKey], cachedResources)) {
          _resolvedAssetResources.remove(assetKey);
        }
        rethrow;
      }
    }
    final resourcesFuture = _assetResources[assetKey] ??= () async {
      final library = await gpu.ShaderLibrary.fromAsset(assetKey);
      final vertexShader = library?['GeometryVertex'];
      final fragmentShader = library?['GeometryFragment'];
      final materialGradientFragmentShader =
          library?['MaterialGradientFragment'];
      final materialTintGradientFragmentShader =
          library?['MaterialTintGradientFragment'];
      final copyFragmentShader = library?['GeometryCopyFragment'];
      if (vertexShader == null ||
          fragmentShader == null ||
          materialGradientFragmentShader == null ||
          materialTintGradientFragmentShader == null ||
          copyFragmentShader == null) {
        throw StateError(
          'LiquidGlass requires Flutter GPU. Run with Flutter 3.47 or newer '
          'and enable Flutter GPU for the target platform.',
        );
      }
      return _SharedGeometryResources(
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        materialGradientFragmentShader: materialGradientFragmentShader,
        materialTintGradientFragmentShader: materialTintGradientFragmentShader,
        copyFragmentShader: copyFragmentShader,
      );
    }();
    try {
      final resources = await resourcesFuture;
      final renderer = FlutterGpuGeometryRenderer._fromShared(resources);
      _resolvedAssetResources[assetKey] = resources;
      if (identical(_assetResources[assetKey], resourcesFuture)) {
        unawaited(_assetResources.remove(assetKey));
      }
      return renderer;
    } on Object {
      // Forget the failed load so a later layer retries it instead of
      // awaiting the same failed Future.
      if (identical(_assetResources[assetKey], resourcesFuture)) {
        unawaited(_assetResources.remove(assetKey));
      }
      rethrow;
    }
  }

  /// Synchronously builds a renderer from already resolved shared resources.
  ///
  /// Returns null when [fromAsset] has not resolved [assetKey] yet, so callers
  /// can fall back to the asynchronous path. On failure the poisoned cache
  /// entry is evicted exactly like [fromAsset] does.
  static FlutterGpuGeometryRenderer? tryCreateCached(String assetKey) {
    final cachedResources = _resolvedAssetResources[assetKey];
    if (cachedResources == null) return null;
    try {
      return FlutterGpuGeometryRenderer._fromShared(cachedResources);
    } on Object {
      if (identical(_resolvedAssetResources[assetKey], cachedResources)) {
        _resolvedAssetResources.remove(assetKey);
      }
      return null;
    }
  }

  /// Completes once the Flutter GPU context exists without blocking.
  ///
  /// Workaround for Flutter 3.47.1 on Android: the engine creates the
  /// Impeller context on the raster thread after startup, off the critical
  /// path (`shell/common/shell.cc`), and reading `gpu.gpuContext` before
  /// then blocks the UI thread until it exists, which takes 100 ms or more on
  /// some Vulkan devices. The first rasterized frame needs the context, so
  /// waiting for it avoids the block. This initializes the widgets binding
  /// if needed, including before `runApp`. Remove it once the engine creates
  /// the context before the first frame or offers a non-blocking way to wait
  /// for it. On every other platform it completes immediately.
  static Future<void> waitUntilGpuContextAvailable() async {
    if (!Platform.isAndroid) return;
    await WidgetsFlutterBinding.ensureInitialized()
        .waitUntilFirstFrameRasterized;
  }

  static final Map<String, Future<_SharedGeometryResources>> _assetResources =
      {};
  static final Map<String, _SharedGeometryResources> _resolvedAssetResources =
      {};

  /// One bump allocator for every geometry pass in the current frame.
  ///
  /// A HostBuffer retains four device-buffer blocks, so one per renderer
  /// would cost four blocks per layer. A single scratch sized for a frame of
  /// layers keeps that off the native heap.
  static gpu.HostBuffer? _sharedHostBuffer;
  static int _sharedHostBufferBlockLength = 0;
  static Duration? _sharedHostBufferFrame;

  /// The host buffer's frame slot that emplacements currently go into.
  static int _hostBufferSlot = 0;

  /// Command buffers per host buffer frame slot that read its uniforms and
  /// have not completed on the GPU.
  static List<int> _pendingPerSlot = const [];

  /// Emplacements in the current slot's block since it was reset.
  static int _slotEmplacements = 0;

  static const int _hostBufferSlotsPerFrame = 32;

  /// Starts a fresh host buffer with blocks of [blockLength]. Command
  /// buffers in flight keep the old buffer's blocks alive, and their
  /// completions no longer count against the new one's slots.
  static gpu.HostBuffer _createHostBuffer(int blockLength) {
    final hostBuffer = _sharedHostBuffer = gpu.gpuContext.createHostBuffer(
      blockLengthInBytes: blockLength,
    );
    _sharedHostBufferBlockLength = blockLength;
    _hostBufferSlot = 0;
    _slotEmplacements = 0;
    _pendingPerSlot = List.filled(hostBuffer.frameCount, 0);
    return hostBuffer;
  }

  // These ownership counters exclude clones held by callers and native
  // references retained by submitted scenes; they are not GPU memory metrics.
  static int _debugActiveRendererCount = 0;
  static int _debugActiveGeometryTextureCount = 0;
  static int _debugActiveMaterialTextureCount = 0;
  static int _debugTotalRenderCount = 0;

  /// Counts SDF submissions across all owners, including temporary renderers.
  @visibleForTesting
  static int get debugTotalRenderCount => _debugTotalRenderCount;

  @visibleForTesting
  static int get debugActiveRendererCount => _debugActiveRendererCount;

  @visibleForTesting
  static int get debugActiveGeometryTextureCount =>
      _debugActiveGeometryTextureCount;

  @visibleForTesting
  static int get debugActiveMaterialTextureCount =>
      _debugActiveMaterialTextureCount;

  static gpu.HostBuffer _hostBufferForUniformSize(int uniformSize) {
    final alignment = gpu.gpuContext.minimumUniformByteAlignment;
    final alignedSize =
        ((uniformSize + alignment - 1) ~/ alignment) * alignment;
    final blockLength = alignedSize * _hostBufferSlotsPerFrame;
    if (_sharedHostBuffer == null ||
        _sharedHostBufferBlockLength < blockLength) {
      _createHostBuffer(blockLength);
    }
    // This is also defined for direct renderer callers outside
    // handleDrawFrame while still advancing once per engine frame in
    // production.
    final timestamp = SchedulerBinding.instance.currentSystemFrameTimeStamp;
    if (_sharedHostBufferFrame != timestamp) {
      // A reset rewinds into the slot used frameCount resets ago, and
      // emplacements overwrite its blocks from the CPU right away. While
      // the GPU still has to read them, emplacements keep appending to the
      // current slot, and the next call tries again. An overwritten pass
      // would draw another frame's shapes, which an incremental update then
      // keeps.
      final next = (_hostBufferSlot + 1) % _pendingPerSlot.length;
      if (_pendingPerSlot[next] == 0) {
        _sharedHostBuffer!.reset();
        _hostBufferSlot = next;
        _slotEmplacements = 0;
        _sharedHostBufferFrame = timestamp;
      } else {
        assert(() {
          debugDeferredHostBufferResetCount++;
          return true;
        }(), 'Count deferred host buffer resets in debug builds.');
      }
    }
    // Every call is followed by one emplacement of at most the aligned
    // size, so a block holds exactly this many. Past it, HostBuffer would
    // take its overflow path, which binds the whole block (more than the
    // 64 KB uniform range many Vulkan GPUs allow) and allocates another
    // block on every overflow without reusing them.
    if (_slotEmplacements >= _hostBufferSlotsPerFrame) {
      assert(() {
        debugHostBufferReplacementCount++;
        return true;
      }(), 'Count host buffer replacements in debug builds.');
      _createHostBuffer(_sharedHostBufferBlockLength);
      _sharedHostBufferFrame = timestamp;
    }
    _slotEmplacements++;
    return _sharedHostBuffer!;
  }

  /// Host buffers started afresh because the current slot's block was full.
  @visibleForTesting
  static int debugHostBufferReplacementCount = 0;

  /// Host buffer resets put off because the GPU had not finished with the
  /// slot they would rewind into.
  @visibleForTesting
  static int debugDeferredHostBufferResetCount = 0;

  /// Command buffers that read the shared host buffer and have not
  /// completed on the GPU.
  @visibleForTesting
  static int get debugPendingHostBufferReaders =>
      _pendingPerSlot.fold(0, (sum, count) => sum + count);

  late final gpu.RenderPipeline _pipeline;
  late final gpu.RenderPipeline _materialGradientPipeline;
  late final gpu.RenderPipeline _materialTintGradientPipeline;
  late final gpu.RenderPipeline _copyPipeline;
  late final gpu.UniformSlot _copyUniformSlot;
  late final gpu.UniformSlot _copyTextureSlot;
  late final int _copyUniformSize;

  @visibleForTesting
  Object get debugPipelineIdentity => _pipeline;

  @visibleForTesting
  int get debugHostBufferBlockLength => _sharedHostBufferBlockLength;

  @visibleForTesting
  Object? get debugHostBufferIdentity => _sharedHostBuffer;

  /// Number of geometry command buffers submitted by this renderer.
  @visibleForTesting
  int debugRenderCount = 0;

  @visibleForTesting
  bool get debugDisposed => _disposed;

  bool _disposed = false;

  // Uniform slot reflection.
  late final gpu.UniformSlot _uniformSlot;
  late final int _uniformSize;
  late final int _offsetUOffset;
  late final int _offsetUTextureSize;
  late final int _offsetOpticalProps;
  late final int _offsetContourProps;
  late final int _offsetWobble;
  late final int _offsetShapeData;
  late final int _offsetRseData;
  late final int _offsetShapeTints;
  late final int _offsetShapeResponses;
  late final int _offsetShapeBounds;
  late final ByteData _uniformData;
  int _writtenShapeFloats = 0;
  int _writtenRseFloats = 0;

  // Latest matte. Older scenes retain their own native references.
  gpu.Texture? _texture;
  ui.Image? _image;
  gpu.RenderTarget? _renderTarget;
  gpu.Texture? _materialTexture;
  ui.Image? _materialImage;
  gpu.RenderTarget? _materialRenderTarget;

  // Appearance blending is decorative and only visible while shapes merge.
  // One sample per 8x8 full-resolution block makes its SDF work 64x smaller.
  static const int materialRasterScale = 8;

  late final gpu.BufferView _vertexBufferView;

  /// Low-resolution contributor map from the latest appearance render.
  ///
  /// The map fills the top-left sub-rect of the image; its size follows from
  /// the matte size, as in the final shader.
  ui.Image? get materialImage => _materialImage;

  final _TextureRing _mattes = _TextureRing();
  final _TextureRing _materials = _TextureRing();

  /// Renders a new geometry matte and returns it as a [ui.Image].
  ///
  /// The matte fills the top-left `width` x `height` of the image, which is
  /// `textureWidth` x `textureHeight`. The returned image is a non-owning
  /// wrapper — do NOT dispose it. Its texture is written again
  /// [reuseAfterFrames] frames after a later render replaces it.
  ({
    ui.Image image,
    int width,
    int height,
    int textureWidth,
    int textureHeight,
  })
  render({
    required int width,
    required int height,
    required List<double> shapeData,
    required int numShapes,
    required double refractionHeight,
    required double refractionAmount,
    required double offsetX,
    required double offsetY,
    double? edgeDistanceRange,
    bool refractionFitsShape = true,
    double contourExtent = 0.5,
    bool writeMaterials = false,
    bool writeTintOnly = false,
    List<double> appearanceData = const <double>[],
    List<double> rseData = const <double>[],
    List<double> boundsData = const <double>[],
    double wobble = 0,
  }) {
    assert(() {
      debugRenderCount++;
      _debugTotalRenderCount++;
      return true;
    }(), 'Track geometry submissions in debug builds.');
    final matteWidth = _bucketDimension(width);
    final matteHeight = _bucketDimension(height);
    final (viewWidth, viewHeight) = _viewCapacity();

    _ensureFrameCounter();
    _lastRenderFrame = _completedFrames;

    _packUniformData(
      offsetX: offsetX,
      offsetY: offsetY,
      textureWidth: matteWidth.toDouble(),
      textureHeight: matteHeight.toDouble(),
      refractionHeight: refractionHeight,
      refractionAmount: refractionAmount,
      edgeDistanceRange: edgeDistanceRange ?? math.max(12, refractionHeight),
      refractionFitsShape: refractionFitsShape,
      contourExtent: contourExtent,
      materialScale: writeMaterials ? materialRasterScale.toDouble() : 1.0,
      materialMapWidth: writeMaterials
          ? math
                .max(
                  1,
                  (matteWidth + materialRasterScale - 1) ~/ materialRasterScale,
                )
                .toDouble()
          : 1.0,
      materialMapHeight: writeMaterials
          ? math
                .max(
                  1,
                  (matteHeight + materialRasterScale - 1) ~/
                      materialRasterScale,
                )
                .toDouble()
          : 1.0,
      geometryAaHalfWidth: _geometryAaHalfWidth,
      numShapes: numShapes.toDouble(),
      shapeData: shapeData,
      rseData: rseData,
      appearanceData: appearanceData,
      boundsData: boundsData,
      wobble: wobble,
    );

    // Most renders change a few shapes. The matte they leave is copied and
    // only the regions those shapes can reach are drawn again; with no shape
    // changed, the copy is all. The copy goes into the next texture of the
    // ring, never into the matte in use: scenes in flight may still sample
    // it, and new textures follow the same order as a full render.
    final previous = _lastMatte;
    final update = previous != null && identical(previous.texture, _texture)
        ? _planMatteUpdate(previous, matteWidth, matteHeight, numShapes)
        : null;

    if (_texture != null) {
      // Release our handle; earlier submitted scenes hold their own.
      _image?.dispose();
      _image = null;
      _renderTarget = null;
      _texture = null;
      assert(() {
        _debugActiveGeometryTextureCount--;
        return true;
      }(), 'Track replaced geometry textures in debug builds.');
    }
    // The passes write every pixel of the sub-rect; the rest of the texture
    // is undefined and never sampled, so no clear is needed.
    final matte = _mattes.next(
      _completedFrames,
      width: matteWidth,
      height: matteHeight,
      maxWidth: viewWidth,
      maxHeight: viewHeight,
    );
    _texture = matte.texture;
    _renderTarget = matte.renderTarget;
    _image = _texture!.asImage();
    assert(() {
      _debugActiveGeometryTextureCount++;
      return true;
    }(), 'Track live geometry textures in debug builds.');

    final materialMapWidth = math.max(
      1,
      (matteWidth + materialRasterScale - 1) ~/ materialRasterScale,
    );
    final materialMapHeight = math.max(
      1,
      (matteHeight + materialRasterScale - 1) ~/ materialRasterScale,
    );
    final materialWidth = writeTintOnly
        ? materialMapWidth
        : math.max(16, materialMapWidth);
    final materialHeight = writeTintOnly
        ? materialMapHeight
        : materialMapHeight + 2;
    if (_materialTexture != null) {
      _materialImage?.dispose();
      _materialImage = null;
      _materialRenderTarget = null;
      _materialTexture = null;
      assert(() {
        _debugActiveMaterialTextureCount--;
        return true;
      }(), 'Track replaced material textures in debug builds.');
    }
    if (writeMaterials) {
      final material = _materials.next(
        _completedFrames,
        width: materialWidth,
        height: materialHeight,
        maxWidth: math.max(
          16,
          (viewWidth + materialRasterScale - 1) ~/ materialRasterScale,
        ),
        maxHeight:
            (viewHeight + materialRasterScale - 1) ~/ materialRasterScale + 2,
      );
      _materialTexture = material.texture;
      _materialRenderTarget = material.renderTarget;
      _materialImage = _materialTexture!.asImage();
      assert(() {
        _debugActiveMaterialTextureCount++;
        return true;
      }(), 'Track live material textures in debug builds.');
    } else {
      _materials.releaseCurrent(_completedFrames);
    }
    if (_mattes.spareCount + _materials.spareCount > 0) {
      _renderersWithSpares.add(this);
    }

    final uniformView = _hostBufferForUniformSize(
      _uniformSize,
    ).emplace(_uniformData);

    final geometryCommandBuffer = gpu.gpuContext.createCommandBuffer();
    final geometryPass = geometryCommandBuffer.createRenderPass(
      _renderTarget!,
    );
    if (update == null) {
      geometryPass
        ..bindPipeline(_pipeline)
        ..setPrimitiveType(gpu.PrimitiveType.triangleStrip)
        ..bindUniform(_uniformSlot, uniformView)
        ..bindVertexBuffer(_vertexBufferView);
      _restrictTo(geometryPass, _texture!, matteWidth, matteHeight);
      geometryPass.draw(4);
      assert(() {
        debugLastRedrawnPixels = matteWidth * matteHeight;
        return true;
      }(), 'Track redrawn matte pixels in debug builds.');
    } else {
      _drawMatteUpdate(
        geometryPass,
        previous!,
        update,
        uniformView,
        matteWidth,
        matteHeight,
      );
    }
    _submitOrDefer(geometryCommandBuffer);
    _lastMatte = (
      texture: _texture!,
      width: matteWidth,
      height: matteHeight,
      uniforms: Float32List.fromList(_uniformData.buffer.asFloat32List()),
    );
    if (writeMaterials) {
      final materialCommandBuffer = gpu.gpuContext.createCommandBuffer();
      final materialPass =
          materialCommandBuffer.createRenderPass(_materialRenderTarget!)
            ..bindPipeline(
              writeTintOnly
                  ? _materialTintGradientPipeline
                  : _materialGradientPipeline,
            )
            ..setPrimitiveType(gpu.PrimitiveType.triangleStrip)
            ..bindUniform(_uniformSlot, uniformView)
            ..bindVertexBuffer(_vertexBufferView);
      _restrictTo(
        materialPass,
        _materialTexture!,
        materialWidth,
        materialHeight,
      );
      materialPass.draw(4);
      _submitOrDefer(materialCommandBuffer);
    }

    return (
      image: _image!,
      width: matteWidth,
      height: matteHeight,
      textureWidth: _texture!.width,
      textureHeight: _texture!.height,
    );
  }

  /// The matte the last render drew, with the uniforms it drew it from.
  _DrawnMatte? _lastMatte;

  /// Whether a render may update the previous matte instead of drawing it
  /// whole: it copies the matte and draws again only around the shapes that
  /// changed. On by default; the result is the same either way.
  static bool incrementalMatteUpdates = true;

  /// Renders that updated the previous matte instead of drawing it whole.
  @visibleForTesting
  int debugIncrementalRenderCount = 0;

  /// Matte pixels the geometry pass drew in the last render.
  @visibleForTesting
  int debugLastRedrawnPixels = 0;

  /// Once the regions to draw again cover this share of the matte, the whole
  /// matte is drawn instead.
  static const double _maxRedrawnShare = 0.6;

  /// Plans how to bring [previous] up to the uniforms just packed, or returns
  /// null when the whole matte must be drawn.
  ///
  /// A pixel depends only on the shapes whose boxes lie within the contour
  /// extent plus three blend radii of its group (see `sceneSample` in
  /// sdf.glsl), so only that reach around a changed shape's old and new box
  /// is drawn again. Everything else must match, apart from a whole-pixel
  /// move of the matte, whose uncovered edges are drawn again too.
  _MatteUpdate? _planMatteUpdate(
    _DrawnMatte previous,
    int width,
    int height,
    int numShapes,
  ) {
    if (!incrementalMatteUpdates) return null;
    final old = previous.uniforms;
    final now = _uniformData.buffer.asFloat32List();
    bool same(int index, int count) {
      for (var i = index; i < index + count; i++) {
        if (old[i] != now[i]) return false;
      }
      return true;
    }

    // Optics, shape count, contour extent and wobble, which every pixel
    // depends on.
    final contourIndex = _offsetContourProps ~/ 4;
    if (!same(_offsetUTextureSize ~/ 4, 2) ||
        !same(_offsetOpticalProps ~/ 4, 4) ||
        old[contourIndex] != now[contourIndex] ||
        old[_offsetWobble ~/ 4] != now[_offsetWobble ~/ 4]) {
      return null;
    }
    final offsetIndex = _offsetUOffset ~/ 4;
    final offsetX = now[offsetIndex];
    final offsetY = now[offsetIndex + 1];
    final shiftX = offsetX - old[offsetIndex];
    final shiftY = offsetY - old[offsetIndex + 1];
    if (shiftX != 0 || shiftY != 0) {
      // Pixel centers land on the same points only for whole-pixel offsets.
      bool whole(double value) => value == value.roundToDouble();
      if (!whole(offsetX) ||
          !whole(offsetY) ||
          !whole(old[offsetIndex]) ||
          !whole(old[offsetIndex + 1])) {
        return null;
      }
    }

    final rects = <_PixelRect>[];
    void add(double left, double top, double right, double bottom) {
      final rect = _PixelRect(
        left.floor().clamp(0, width),
        top.floor().clamp(0, height),
        right.ceil().clamp(0, width),
        bottom.ceil().clamp(0, height),
      );
      if (!rect.isEmpty) rects.add(rect);
    }

    final shapeIndex = _offsetShapeData ~/ 4;
    final rseIndex = _offsetRseData ~/ 4;
    final boundsIndex = _offsetShapeBounds ~/ 4;
    final contourReach = now[contourIndex] + 4;
    for (var i = 0; i < math.min(numShapes, 16); i++) {
      final shape = shapeIndex + i * 12;
      // A changed marker regroups shapes or changes a blend radius.
      if (old[shape + 11] != now[shape + 11]) return null;
      final bounds = boundsIndex + i * 4;
      if (same(shape, 12) && same(rseIndex + i * 12, 12) && same(bounds, 4)) {
        continue;
      }
      // A shape alone in its group blends with nothing.
      var start = i;
      while (start > 0 && now[shapeIndex + start * 12 + 11] >= 0) {
        start--;
      }
      final alone =
          i == start &&
          (i + 1 >= numShapes || now[shapeIndex + (i + 1) * 12 + 11] < 0);
      final marker = now[shapeIndex + start * 12 + 11];
      final blend = alone ? 0 : (marker < 0 ? -marker - 1 : marker);
      final reach = contourReach + 3 * blend;
      add(
        math.min(old[bounds], now[bounds]) - reach - offsetX,
        math.min(old[bounds + 1], now[bounds + 1]) - reach - offsetY,
        math.max(old[bounds + 2], now[bounds + 2]) + reach - offsetX,
        math.max(old[bounds + 3], now[bounds + 3]) + reach - offsetY,
      );
    }
    // Pixels the previous matte did not cover.
    final dx = shiftX.toInt();
    final dy = shiftY.toInt();
    final coveredLeft = -dx;
    final coveredTop = -dy;
    final coveredRight = previous.width - dx;
    final coveredBottom = previous.height - dy;
    add(0, 0, coveredLeft.toDouble(), height.toDouble());
    add(coveredRight.toDouble(), 0, width.toDouble(), height.toDouble());
    add(0, 0, width.toDouble(), coveredTop.toDouble());
    add(0, coveredBottom.toDouble(), width.toDouble(), height.toDouble());

    final merged = _PixelRect.merge(rects);
    final redrawn = merged.fold(0, (sum, rect) => sum + rect.area);
    if (redrawn >= width * height * _maxRedrawnShare) return null;
    return _MatteUpdate(shiftX: dx, shiftY: dy, rects: merged);
  }

  /// Copies [previous] into the new matte and draws [update]'s regions
  /// again.
  void _drawMatteUpdate(
    gpu.RenderPass pass,
    _DrawnMatte previous,
    _MatteUpdate update,
    gpu.BufferView uniformView,
    int width,
    int height,
  ) {
    // Padded to the geometry uniforms' size: the shared host buffer's
    // blocks hold whole geometry slots, and a shorter write would leave the
    // next geometry slot straddling a block's end.
    final copyData = Float32List(math.max(_uniformSize, _copyUniformSize) ~/ 4)
      ..[0] = update.shiftX.toDouble()
      ..[1] = update.shiftY.toDouble()
      ..[2] = previous.texture.width.toDouble()
      ..[3] = previous.texture.height.toDouble();
    final copyView = _hostBufferForUniformSize(
      _uniformSize,
    ).emplace(ByteData.sublistView(copyData));
    final viewport = gpu.Viewport(width: width, height: height);
    pass
      ..bindPipeline(_copyPipeline)
      ..setPrimitiveType(gpu.PrimitiveType.triangleStrip)
      ..bindUniform(_copyUniformSlot, copyView)
      ..bindTexture(_copyTextureSlot, previous.texture)
      ..bindVertexBuffer(_vertexBufferView)
      ..setViewport(viewport)
      ..setScissor(gpu.Scissor(width: width, height: height))
      ..draw(4);
    var redrawn = 0;
    for (final rect in update.rects) {
      // Bindings outlive a draw in Flutter GPU (3.47.1). The copy's texture
      // left bound for the geometry pipeline, which has no sampler, crashes
      // the Vulkan backend at submission (flutter_tester on SwiftShader:
      // segmentation fault), so each draw starts from clear bindings.
      pass
        ..clearBindings()
        ..bindPipeline(_pipeline)
        ..setPrimitiveType(gpu.PrimitiveType.triangleStrip)
        ..bindUniform(_uniformSlot, uniformView)
        ..bindVertexBuffer(_vertexBufferView)
        ..setViewport(viewport)
        ..setScissor(
          gpu.Scissor(
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height,
          ),
        )
        ..draw(4);
      redrawn += rect.area;
    }
    assert(() {
      debugIncrementalRenderCount++;
      debugLastRedrawnPixels = redrawn;
      return true;
    }(), 'Count incremental matte updates in debug builds.');
  }

  /// Limits [pass] to the top-left [width] x [height] of [texture].
  ///
  /// `gl_FragCoord` stays framebuffer-relative, so the shaders need no
  /// offset for a sub-rect at the origin.
  static void _restrictTo(
    gpu.RenderPass pass,
    gpu.Texture texture,
    int width,
    int height,
  ) {
    if (texture.width == width && texture.height == height) return;
    pass
      ..setViewport(gpu.Viewport(width: width, height: height))
      ..setScissor(gpu.Scissor(width: width, height: height));
  }

  /// Largest view in physical pixels, the cap for texture growth.
  static (int, int) _viewCapacity() {
    var width = 0;
    var height = 0;
    for (final view in WidgetsBinding.instance.platformDispatcher.views) {
      width = math.max(width, view.physicalSize.width.ceil());
      height = math.max(height, view.physicalSize.height.ceil());
    }
    return (_bucketDimension(width), _bucketDimension(height));
  }

  static int _bucketDimension(int value) => (value + 63) & ~63;

  // Every pass gets its own command buffer. Flutter GPU (3.47.1) begins the
  // backend pass in `createRenderPass` and ends it only in `submit`, so a
  // second pass on the same command buffer nests inside the first:
  //
  // - Vulkan (Pixel 10, PowerVR; also flutter_tester on SwiftShader):
  //     Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x60
  //     #00 vulkan.powervr.so (CmdEndRenderPass2+188)
  //     #04 libflutter.so (InternalFlutterGpu_CommandBuffer_Submit+56)
  // - Metal (macOS, AGX G16X):
  //     -[AGXG16XFamilyCommandBuffer renderCommandEncoderWithDescriptor:]:
  //     failed assertion `A command encoder is already encoding to this
  //     command buffer'
  //       impeller::RenderPassMTL::RenderPassMTL
  //       impeller::CommandBufferMTL::OnCreateRenderPass
  //       flutter::gpu::RenderPass::Begin
  //
  // Share a command buffer across passes only once the engine ends a pass
  // before the next begins.
  static final List<_TrackedCommandBuffer> _pendingCommandBuffers = [];
  static bool _postFrameFlushScheduled = false;

  /// Command buffers submitted by [flushPendingSubmissions].
  @visibleForTesting
  static int debugBatchedSubmitCount = 0;

  /// Flushes left to the post-frame safety net because no glass layer of the
  /// frame's scene flushed first.
  @visibleForTesting
  static int debugPostFrameFlushCount = 0;

  /// Passes whose submission was deferred to [flushPendingSubmissions].
  @visibleForTesting
  static int debugDeferredPassCount = 0;

  // Only the paint and compositing phases are followed by a scene build that
  // flushes before the scene reaches the raster thread.
  static bool get _deferring =>
      SchedulerBinding.instance.schedulerPhase ==
      SchedulerPhase.persistentCallbacks;

  static void _submitOrDefer(gpu.CommandBuffer commandBuffer) {
    // The command buffer reads uniforms from the current host buffer slot,
    // which must not be reset into until the GPU has run it.
    final slot = _hostBufferSlot;
    final pending = _pendingPerSlot;
    pending[slot]++;
    final tracked = _TrackedCommandBuffer(commandBuffer, () {
      // A host buffer replaced since keeps its own blocks.
      if (identical(pending, _pendingPerSlot)) pending[slot]--;
    });
    if (!_deferring) {
      // An incremental update copies the previous matte, whose pass may
      // still be deferred: submit that first so the queue keeps their order.
      flushPendingSubmissions();
      tracked.submit();
      return;
    }
    assert(() {
      debugDeferredPassCount++;
      return true;
    }(), 'Count deferred geometry passes in debug builds.');
    _pendingCommandBuffers.add(tracked);
    if (_postFrameFlushScheduled) return;
    _postFrameFlushScheduled = true;
    // Covers passes whose layer was painted but not composited.
    SchedulerBinding.instance.addPostFrameCallback((_) {
      _postFrameFlushScheduled = false;
      if (_pendingCommandBuffers.isEmpty) return;
      assert(() {
        debugPostFrameFlushCount++;
        return true;
      }(), 'Count safety-net flushes in debug builds.');
      flushPendingSubmissions();
    });
  }

  /// Submits the passes recorded since the last flush, in recording order.
  ///
  /// Glass layers call this while the scene is built, which is before the
  /// scene is handed to the raster thread, so every matte a scene samples has
  /// been submitted ahead of it on the GPU queue. A post-frame callback
  /// flushes passes of layers that were painted but not composited.
  static void flushPendingSubmissions() {
    if (_pendingCommandBuffers.isEmpty) return;
    // Take the list first: a submit that throws must not leave it holding
    // buffers already submitted, which every later flush would submit again
    // while the ones after it never run.
    final commandBuffers = List.of(_pendingCommandBuffers);
    _pendingCommandBuffers.clear();
    Object? error;
    StackTrace? stackTrace;
    for (final commandBuffer in commandBuffers) {
      try {
        commandBuffer.submit();
      } on Object catch (e, s) {
        error ??= e;
        stackTrace ??= s;
      }
      assert(() {
        debugBatchedSubmitCount++;
        return true;
      }(), 'Count batched submissions in debug builds.');
    }
    if (error != null) Error.throwWithStackTrace(error, stackTrace!);
  }

  /// Frames after its replacement before a texture may be rendered into
  /// again.
  ///
  /// A texture replaced during frame E was last sampled by frame E - 1's
  /// scene. The engine keeps at most two frames in its pipeline and frees a
  /// slot only after rasterizing and submitting that frame (`Animator`'s
  /// `FramePipeline(2)`; `Pipeline::Consume` signals after the consumer
  /// returns), so when frame E + 1 paints, frame E - 1 has been submitted.
  /// Flutter GPU submits to the same queue as the raster thread (the IO
  /// manager's context is the platform view's), and the GPU orders a write
  /// after earlier reads on that queue: Metal tracks hazards on these
  /// non-heap textures, and every Impeller Vulkan render pass declares an
  /// external dependency on earlier fragment-shader reads. GLES submits on
  /// the raster thread in posting order.
  static const int reuseAfterFrames = 1;

  /// Spare textures are dropped after this many frames without a render,
  /// and released ones after this many frames unclaimed.
  static const int _idleFramesBeforeTrim = 120;

  /// Released textures kept for other renderers, app-wide.
  static const int _maxReleased = 4;

  static int _completedFrames = 0;
  static bool _countingFrames = false;
  static final Set<FlutterGpuGeometryRenderer> _renderersWithSpares = {};
  static final List<_RingTexture> _released = [];

  /// Counts frames that submit a scene, the unit [reuseAfterFrames] is in.
  ///
  /// Renders outside a frame, as in unit tests, do not advance the count,
  /// so their textures are never reused.
  static void _ensureFrameCounter() {
    if (_countingFrames) return;
    _countingFrames = true;
    SchedulerBinding.instance.addPersistentFrameCallback((_) {
      if (!RendererBinding.instance.sendFramesToEngine) return;
      _completedFrames++;
      _released.removeWhere((texture) {
        final stale =
            _completedFrames - texture.retiredFrame > _idleFramesBeforeTrim;
        if (stale) _countDropped();
        return stale;
      });
      if (_renderersWithSpares.isEmpty) return;
      for (final renderer in _renderersWithSpares.toList()) {
        if (_completedFrames - renderer._lastRenderFrame >
            _idleFramesBeforeTrim) {
          renderer._mattes.dropSpares();
          renderer._materials.dropSpares();
          _renderersWithSpares.remove(renderer);
        }
      }
    });
  }

  int _lastRenderFrame = 0;

  /// Textures this renderer holds that are not in use.
  @visibleForTesting
  int get debugRetiredTextureCount =>
      _mattes.spareCount + _materials.spareCount;

  /// Matte textures this renderer holds, including the one in use.
  @visibleForTesting
  int get debugMatteTextureCount => _mattes.textureCount;

  /// The latest matte, read back in tests.
  @visibleForTesting
  ui.Image? get debugMatteImage => _image;

  /// The latest matte texture, compared by identity in tests.
  @visibleForTesting
  Object? get debugMatteTexture => _texture;

  /// Size of the latest matte texture.
  @visibleForTesting
  (int, int)? get debugMatteTextureSize => switch (_texture) {
    final texture? => (texture.width, texture.height),
    null => null,
  };

  /// Renders that wrote into a reused texture instead of allocating one.
  @visibleForTesting
  static int debugReusedTextureCount = 0;

  /// Textures allocated, all renderers.
  @visibleForTesting
  static int debugAllocatedTextureCount = 0;

  /// Textures dropped, all renderers. Once a texture has lived a few young
  /// GCs its wrapper is promoted, and dropping it pins its storage until an
  /// old-generation GC.
  @visibleForTesting
  static int debugDroppedTextureCount = 0;

  /// Released textures waiting for another renderer.
  @visibleForTesting
  static int get debugReleasedTextureCount => _released.length;

  static void _countDropped() {
    assert(() {
      debugDroppedTextureCount++;
      return true;
    }(), 'Count dropped geometry textures in debug builds.');
  }

  /// A mature released texture that holds [width] x [height] without
  /// wasting more than 1.5x per dimension, or null.
  static _RingTexture? _claimReleased(int frame, int width, int height) {
    for (var index = 0; index < _released.length; index++) {
      final texture = _released[index];
      if (frame - texture.retiredFrame < reuseAfterFrames) continue;
      if (texture.covers(width, height) &&
          texture.texture.width * texture.texture.height * 4 <=
              width * height * 9) {
        return _released.removeAt(index);
      }
    }
    return null;
  }

  void _packUniformData({
    required double offsetX,
    required double offsetY,
    required double textureWidth,
    required double textureHeight,
    required double refractionHeight,
    required double refractionAmount,
    required double edgeDistanceRange,
    required bool refractionFitsShape,
    required double contourExtent,
    required double materialScale,
    required double materialMapWidth,
    required double materialMapHeight,
    required double geometryAaHalfWidth,
    required double numShapes,
    required List<double> shapeData,
    required List<double> rseData,
    required List<double> appearanceData,
    required List<double> boundsData,
    required double wobble,
  }) {
    final floatData = _uniformData.buffer.asFloat32List();
    floatData[_offsetWobble ~/ 4] = math.max(0, wobble);

    final uOffsetIndex = _offsetUOffset ~/ 4;
    floatData[uOffsetIndex] = offsetX;
    floatData[uOffsetIndex + 1] = offsetY;

    final textureSizeIndex = _offsetUTextureSize ~/ 4;
    // X selects shape-fitted refraction; Y is the bevel's edge displacement,
    // which is also the codec scale.
    floatData[textureSizeIndex] = refractionFitsShape ? 1 : 0;
    floatData[textureSizeIndex + 1] = math.max(1e-3, refractionAmount);

    final opticalPropsIndex = _offsetOpticalProps ~/ 4;
    floatData[opticalPropsIndex] = math.max(0, refractionHeight);
    // uOpticalProps.y is the centered-AA half-width: 0.5 (Flutter's
    // one-pixel transition) unless the harness overrides it.
    floatData[opticalPropsIndex + 1] = geometryAaHalfWidth.clamp(0.0, 1.0);
    floatData[opticalPropsIndex + 2] = math.max(1, edgeDistanceRange);
    floatData[opticalPropsIndex + 3] = numShapes;

    final contourPropsIndex = _offsetContourProps ~/ 4;
    floatData[contourPropsIndex] = math.max(0.5, contourExtent);
    floatData[contourPropsIndex + 1] = materialScale;
    floatData[contourPropsIndex + 2] = materialMapWidth;
    floatData[contourPropsIndex + 3] = materialMapHeight;

    final shapeDataStartIndex = _offsetShapeData ~/ 4;
    final shapeFloats = shapeData.length < 192 ? shapeData.length : 192;
    for (var i = 0; i < shapeFloats; i++) {
      floatData[shapeDataStartIndex + i] = shapeData[i];
    }
    for (var i = shapeFloats; i < _writtenShapeFloats; i++) {
      floatData[shapeDataStartIndex + i] = 0;
    }
    _writtenShapeFloats = shapeFloats;

    final rseDataStartIndex = _offsetRseData ~/ 4;
    final rseFloats = rseData.length < 192 ? rseData.length : 192;
    for (var i = 0; i < rseFloats; i++) {
      floatData[rseDataStartIndex + i] = rseData[i];
    }
    for (var i = rseFloats; i < _writtenRseFloats; i++) {
      floatData[rseDataStartIndex + i] = 0;
    }
    _writtenRseFloats = rseFloats;

    if (appearanceData.isNotEmpty && appearanceData.length != 16 * 2 * 4) {
      throw ArgumentError.value(
        appearanceData.length,
        'appearanceData.length',
        'must contain two vec4 rows for each of 16 shapes',
      );
    }
    if (appearanceData.isNotEmpty) {
      final shapeTintsStartIndex = _offsetShapeTints ~/ 4;
      final shapeResponsesStartIndex = _offsetShapeResponses ~/ 4;
      for (var i = 0; i < 16 * 4; i++) {
        floatData[shapeTintsStartIndex + i] = appearanceData[i];
        floatData[shapeResponsesStartIndex + i] = appearanceData[16 * 4 + i];
      }
    }

    // Shapes without bounds are never culled.
    final boundsStartIndex = _offsetShapeBounds ~/ 4;
    final boundsFloats = math.min(boundsData.length, 16 * 4);
    for (var i = 0; i < 16 * 4; i++) {
      floatData[boundsStartIndex + i] = i < boundsFloats
          ? boundsData[i]
          : (i % 4 < 2 ? -1e9 : 1e9);
    }
  }

  /// Releases borrowed output handles, keeping reusable rendering resources.
  ///
  /// Callers retaining an output must clone its images before calling this.
  /// Independent clones and submitted native scenes remain valid.
  void releaseOutput() {
    _lastMatte = null;
    _mattes.release(_completedFrames);
    _materials.release(_completedFrames);
    _renderersWithSpares.remove(this);
    _renderTarget = null;
    _image?.dispose();
    _image = null;
    _materialRenderTarget = null;
    _materialImage?.dispose();
    _materialImage = null;
    if (_materialTexture != null) {
      assert(() {
        _debugActiveMaterialTextureCount--;
        return true;
      }(), 'Track disposed material textures in debug builds.');
    }
    _materialTexture = null;
    if (_texture != null) {
      assert(() {
        _debugActiveGeometryTextureCount--;
        return true;
      }(), 'Track disposed geometry textures in debug builds.');
    }
    _texture = null;
  }

  /// Releases the current output and retires this renderer.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    releaseOutput();
    assert(() {
      _debugActiveRendererCount--;
      return true;
    }(), 'Track disposed geometry renderers in debug builds.');
  }
}

/// A texture and its render target.
final class _RingTexture {
  _RingTexture(this.texture)
    : renderTarget = gpu.RenderTarget.singleColor(
        gpu.ColorAttachment(
          texture: texture,
          loadAction: gpu.LoadAction.dontCare,
        ),
      );

  final gpu.Texture texture;
  final gpu.RenderTarget renderTarget;

  /// Frame count when this texture was last replaced or released.
  int retiredFrame = 0;

  bool covers(int width, int height) =>
      texture.width >= width && texture.height >= height;
}

/// The textures one output of a renderer cycles through.
///
/// Every texture is at least the ring's capacity, which only grows: in 1.5x
/// steps, capped at the view unless one render needs more. A resize within
/// the capacity reuses the ring; growth drops the smaller textures.
final class _TextureRing {
  /// Spares beyond the one a steady ring needs, for frames that render
  /// twice.
  static const int _maxSpares = 2;

  _RingTexture? _current;
  final List<_RingTexture> _spares = [];
  int _width = 0;
  int _height = 0;

  int get spareCount => _spares.length;
  int get textureCount => _spares.length + (_current == null ? 0 : 1);

  /// Replaces the texture in use with one that holds [width] x [height].
  _RingTexture next(
    int frame, {
    required int width,
    required int height,
    required int maxWidth,
    required int maxHeight,
  }) {
    final previous = _current;
    _current = null;
    if (width > _width || height > _height) {
      _width = _grow(_width, width, maxWidth);
      _height = _grow(_height, height, maxHeight);
      dropSpares(keep: (texture) => texture.covers(_width, _height));
    }
    if (previous != null) {
      if (previous.covers(_width, _height)) {
        previous.retiredFrame = frame;
        _spares.add(previous);
      } else {
        FlutterGpuGeometryRenderer._countDropped();
      }
    }
    // Spares retire in frame order, so the first is the oldest.
    if (_spares.isNotEmpty &&
        frame - _spares.first.retiredFrame >=
            FlutterGpuGeometryRenderer.reuseAfterFrames) {
      assert(() {
        FlutterGpuGeometryRenderer.debugReusedTextureCount++;
        return true;
      }(), 'Count reused geometry textures in debug builds.');
      return _current = _spares.removeAt(0);
    }
    while (_spares.length > _maxSpares) {
      _spares.removeAt(0);
      FlutterGpuGeometryRenderer._countDropped();
    }
    final released = FlutterGpuGeometryRenderer._claimReleased(
      frame,
      _width,
      _height,
    );
    if (released != null) {
      assert(() {
        FlutterGpuGeometryRenderer.debugReusedTextureCount++;
        return true;
      }(), 'Count reused geometry textures in debug builds.');
      return _current = released;
    }
    return _current = _allocate(_width, _height);
  }

  /// Retires the texture in use without replacing it.
  void releaseCurrent(int frame) {
    final previous = _current;
    _current = null;
    if (previous == null) return;
    previous.retiredFrame = frame;
    _spares.add(previous);
  }

  /// Drops spare textures, except those [keep] accepts.
  void dropSpares({bool Function(_RingTexture texture)? keep}) {
    _spares.removeWhere((texture) {
      if (keep?.call(texture) ?? false) return false;
      FlutterGpuGeometryRenderer._countDropped();
      return true;
    });
  }

  /// Hands every texture to the app-wide released list.
  void release(int frame) {
    final textures = [..._spares, ?_current];
    _spares.clear();
    _current = null;
    _width = 0;
    _height = 0;
    final released = FlutterGpuGeometryRenderer._released;
    for (final texture in textures) {
      texture.retiredFrame = frame;
      released.add(texture);
    }
    while (released.length > FlutterGpuGeometryRenderer._maxReleased) {
      released.removeAt(0);
      FlutterGpuGeometryRenderer._countDropped();
    }
  }

  static int _grow(int capacity, int needed, int limit) {
    if (needed <= capacity) return capacity;
    if (capacity == 0) return needed;
    final stepped = (capacity * 3 + 1) >> 1;
    return math.max(needed, math.min(stepped, limit));
  }

  static _RingTexture _allocate(int width, int height) {
    assert(() {
      FlutterGpuGeometryRenderer.debugAllocatedTextureCount++;
      return true;
    }(), 'Count allocated geometry textures in debug builds.');
    return _RingTexture(
      gpu.gpuContext.createTexture(
        gpu.StorageMode.devicePrivate,
        width,
        height,
      ),
    );
  }
}

class _SharedGeometryResources {
  _SharedGeometryResources({
    required gpu.Shader vertexShader,
    required gpu.Shader fragmentShader,
    required gpu.Shader materialGradientFragmentShader,
    required gpu.Shader materialTintGradientFragmentShader,
    required gpu.Shader copyFragmentShader,
  }) {
    pipeline = gpu.gpuContext.createRenderPipeline(
      vertexShader,
      fragmentShader,
    );
    materialGradientPipeline = gpu.gpuContext.createRenderPipeline(
      vertexShader,
      materialGradientFragmentShader,
    );
    materialTintGradientPipeline = gpu.gpuContext.createRenderPipeline(
      vertexShader,
      materialTintGradientFragmentShader,
    );
    copyPipeline = gpu.gpuContext.createRenderPipeline(
      vertexShader,
      copyFragmentShader,
    );
    copyUniformSlot = copyFragmentShader.getUniformSlot('CopyUniforms');
    copyTextureSlot = copyFragmentShader.getUniformSlot('uPreviousMatte');
    copyUniformSize = copyUniformSlot.sizeInBytes ?? 16;
    uniformSlot = fragmentShader.getUniformSlot('GeometryUniforms');
    uniformSize = uniformSlot.sizeInBytes ?? 0;
    offsetUOffset = uniformSlot.getMemberOffsetInBytes('uOffset') ?? 0;
    offsetUTextureSize =
        uniformSlot.getMemberOffsetInBytes('uTextureSize') ?? 0;
    offsetOpticalProps =
        uniformSlot.getMemberOffsetInBytes('uOpticalProps') ?? 0;
    offsetContourProps =
        uniformSlot.getMemberOffsetInBytes('uContourProps') ?? 0;
    offsetShapeData = uniformSlot.getMemberOffsetInBytes('uShapeData') ?? 0;
    offsetRseData = uniformSlot.getMemberOffsetInBytes('uRseData') ?? 0;
    offsetShapeTints = uniformSlot.getMemberOffsetInBytes('uShapeTints') ?? 0;
    offsetShapeResponses =
        uniformSlot.getMemberOffsetInBytes('uShapeResponses') ?? 0;
    offsetShapeBounds = uniformSlot.getMemberOffsetInBytes('uShapeBounds') ?? 0;
    offsetWobble = uniformSlot.getMemberOffsetInBytes('uWobble') ?? 0;
    final vertices = Float32List.fromList([
      -1.0, -1.0, 0.0, 0.0, //
      1.0, -1.0, 1.0, 0.0, //
      -1.0, 1.0, 0.0, 1.0, //
      1.0, 1.0, 1.0, 1.0, //
    ]);
    vertexBuffer = gpu.gpuContext.createDeviceBufferWithCopy(
      ByteData.sublistView(vertices),
    );
    vertexBufferView = gpu.BufferView(
      vertexBuffer,
      offsetInBytes: 0,
      lengthInBytes: vertexBuffer.sizeInBytes,
    );
  }

  late final gpu.RenderPipeline pipeline;
  late final gpu.RenderPipeline materialGradientPipeline;
  late final gpu.RenderPipeline materialTintGradientPipeline;
  late final gpu.RenderPipeline copyPipeline;
  late final gpu.UniformSlot copyUniformSlot;
  late final gpu.UniformSlot copyTextureSlot;
  late final int copyUniformSize;
  late final gpu.UniformSlot uniformSlot;
  late final int uniformSize;
  late final int offsetUOffset;
  late final int offsetUTextureSize;
  late final int offsetOpticalProps;
  late final int offsetContourProps;
  late final int offsetShapeData;
  late final int offsetRseData;
  late final int offsetShapeTints;
  late final int offsetShapeResponses;
  late final int offsetShapeBounds;
  late final int offsetWobble;
  late final gpu.DeviceBuffer vertexBuffer;
  late final gpu.BufferView vertexBufferView;
}

/// A matte a render drew, and the uniforms it drew it from.
typedef _DrawnMatte = ({
  gpu.Texture texture,
  int width,
  int height,
  Float32List uniforms,
});

/// A command buffer that reports when the GPU has run it.
final class _TrackedCommandBuffer {
  _TrackedCommandBuffer(this._commandBuffer, this._onDone);

  final gpu.CommandBuffer _commandBuffer;
  final void Function() _onDone;
  bool _done = false;

  void submit() {
    try {
      // Completion is posted to the UI thread, failed or not.
      _commandBuffer.submit(completionCallback: (_) => _finish());
    } catch (_) {
      _finish();
      rethrow;
    }
  }

  void _finish() {
    if (_done) return;
    _done = true;
    _onDone();
  }
}

/// How a render brings the previous matte up to date: copy it, moved by
/// whole pixels, and draw [rects] again.
final class _MatteUpdate {
  const _MatteUpdate({
    required this.shiftX,
    required this.shiftY,
    required this.rects,
  });

  /// Offset from a pixel of the new matte to the same point in the previous
  /// one.
  final int shiftX;
  final int shiftY;

  /// Regions of the new matte to draw again, in pixels.
  final List<_PixelRect> rects;
}

/// A rectangle of whole pixels, right and bottom exclusive.
final class _PixelRect {
  const _PixelRect(this.left, this.top, this.right, this.bottom);

  final int left;
  final int top;
  final int right;
  final int bottom;

  int get width => right - left;
  int get height => bottom - top;
  int get area => isEmpty ? 0 : width * height;
  bool get isEmpty => right <= left || bottom <= top;

  _PixelRect union(_PixelRect other) => _PixelRect(
    math.min(left, other.left),
    math.min(top, other.top),
    math.max(right, other.right),
    math.max(bottom, other.bottom),
  );

  /// Merges rects whose union costs no more than drawing them apart, then
  /// the cheapest pairs, until at most [maxRects] remain.
  static List<_PixelRect> merge(List<_PixelRect> rects, {int maxRects = 4}) {
    final result = [...rects];
    while (result.length > 1) {
      var bestI = 0;
      var bestJ = 1;
      var bestWaste = double.infinity;
      for (var i = 0; i < result.length; i++) {
        for (var j = i + 1; j < result.length; j++) {
          final waste =
              result[i].union(result[j]).area - result[i].area - result[j].area;
          if (waste < bestWaste) {
            bestWaste = waste.toDouble();
            bestI = i;
            bestJ = j;
          }
        }
      }
      if (bestWaste > 0 && result.length <= maxRects) break;
      result[bestI] = result[bestI].union(result[bestJ]);
      result.removeAt(bestJ);
    }
    return result;
  }
}
