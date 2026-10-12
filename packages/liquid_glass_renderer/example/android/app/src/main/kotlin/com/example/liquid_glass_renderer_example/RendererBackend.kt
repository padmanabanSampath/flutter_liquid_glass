package com.example.liquid_glass_renderer_example

import android.content.Context
import android.os.Build

/**
 * Chooses Impeller's graphics backend before the engine starts.
 *
 * The backend is an engine flag, read once per process, so it cannot change
 * while the app runs: a new choice applies after a restart. "auto" follows
 * [autoBackend]; "vulkan" and "opengles" force that backend. With no flag the
 * engine picks its default (Vulkan, unless the device cannot run it).
 */
object RendererBackend {
    const val AUTO = "auto"
    const val VULKAN = "vulkan"
    const val OPENGLES = "opengles"

    private const val PREFERENCES = "liquid_glass_renderer"
    private const val KEY = "impellerBackend"

    /**
     * SoCs whose GPUs render the glass more efficiently on OpenGL ES, as
     * reported by Build.HARDWARE or Build.SOC_MODEL. Measured on a Mali-G615
     * MC5 (Dimensity 8300, which tablets report as mt8792; driver r44p1):
     * about 20% less power than Vulkan at the same 60 fps. Extend after
     * measuring other devices.
     */
    private val openGlesSocs = setOf("mt8792", "mt6897")

    /** The saved choice: [AUTO], [VULKAN] or [OPENGLES]. */
    fun preference(context: Context): String =
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(KEY, AUTO) ?: AUTO

    /** Saves [choice] synchronously, so a restart right after keeps it. */
    fun setPreference(context: Context, choice: String) {
        require(choice == AUTO || choice == VULKAN || choice == OPENGLES) {
            "Unknown renderer choice: $choice"
        }
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY, choice)
            .commit()
    }

    /** The backend "auto" picks on this device, or null for the default. */
    fun autoBackend(): String? {
        val socs = buildList {
            add(Build.HARDWARE)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) add(Build.SOC_MODEL)
        }.map { it.lowercase() }
        return if (socs.any { it in openGlesSocs }) OPENGLES else null
    }

    /** The backend to request for this launch, or null for the default. */
    fun forLaunch(context: Context): String? =
        when (val choice = preference(context)) {
            AUTO -> autoBackend()
            else -> choice
        }
}
