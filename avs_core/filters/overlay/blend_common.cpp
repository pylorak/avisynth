// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

// Overlay (c) 2003, 2004 by Klaus Post

#include <avs/config.h>

#include "blend_common.h"
#include "overlayfunctions.h"

// Masked blend
// for blend mode
static BYTE OV_FORCEINLINE overlay_blend_c_core(const BYTE p1, const BYTE p2, const int mask) {
  return (BYTE)((((p1<<8) | 128) + (p2-p1)*mask) >> 8);
}
#ifdef X86_32
static __m64 OV_FORCEINLINE overlay_blend_mmx_core(const __m64& p1, const __m64& p2, const __m64& mask, const __m64& v128) {
  __m64 tmp1 = _mm_mullo_pi16(_mm_sub_pi16(p2, p1), mask); // (p2-p1)*mask
  __m64 tmp2 = _mm_or_si64(_mm_slli_pi16(p1, 8), v128);    // p1<<8 + 128 == p1<<8 | 128
  return _mm_srli_pi16(_mm_add_pi16(tmp1, tmp2), 8);
}
#endif
static __m128i OV_FORCEINLINE overlay_blend_sse2_core(const __m128i& p1, const __m128i& p2, const __m128i& mask, const __m128i& v128) {
  __m128i tmp1 = _mm_mullo_epi16(_mm_sub_epi16(p2, p1), mask); // (p2-p1)*mask
  __m128i tmp2 = _mm_or_si128(_mm_slli_epi16(p1, 8), v128);    // p1<<8 + 128 == p1<<8 | 128
  return _mm_srli_epi16(_mm_add_epi16(tmp1, tmp2), 8);
}

// Merge mask
// Use to combine opacity mask and clip mask
static BYTE OV_FORCEINLINE overley_merge_mask_c(const BYTE p1, const int p2) {
  return (p1*p2) >> 8;
}
#ifdef X86_32
static __m64 OV_FORCEINLINE overlay_merge_mask_mmx(const __m64& p1, const __m64& p2) {
  __m64 t1 = _mm_mullo_pi16(p1, p2);
  __m64 t2 = _mm_srli_pi16(t1, 8);
  return t2;
}
#endif
static __m128i OV_FORCEINLINE overlay_merge_mask_sse2(const __m128i& p1, const __m128i& p2) {
  __m128i t1 = _mm_mullo_epi16(p1, p2);
  __m128i t2 = _mm_srli_epi16(t1, 8);
  return t2;
}

// Blend Opaque
// Used in lighten and darken mode
BYTE OV_FORCEINLINE overlay_blend_opaque_c_core(const BYTE p1, const BYTE p2, const BYTE mask) {
  return (mask) ? p2 : p1;
}
#ifdef X86_32
__m64 OV_FORCEINLINE overlay_blend_opaque_mmx_core(const __m64& p1, const __m64& p2, const __m64& mask) {
  __m64 r1 = _mm_andnot_si64(mask, p1);
  __m64 r2 = _mm_and_si64   (mask, p2);
  return _mm_or_si64(r1, r2);
}
#endif
__m128i OV_FORCEINLINE overlay_blend_opaque_sse2_core(const __m128i& p1, const __m128i& p2, const __m128i& mask) {
  __m128i r1 = _mm_andnot_si128(mask, p1);
  __m128i r2 = _mm_and_si128   (mask, p2);
  return _mm_or_si128(r1, r2);
}

__m128i OV_FORCEINLINE overlay_blend_opaque_sse41_core(const __m128i& p1, const __m128i& p2, const __m128i& mask) {
  return _mm_blendv_epi8(p1, p2, mask);
}

/////////////////////////////////////////////
// Mode: Overlay
/////////////////////////////////////////////

void overlay_blend_c_plane_masked(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                  const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                  const int width, const int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      BYTE result = overlay_blend_c_core(p1[x], p2[x], static_cast<int>(mask[x]));
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}
#ifdef X86_32
#pragma warning (disable: 4799)
void overlay_blend_mmx_plane_masked(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                    const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                    const int width, const int height) {
        BYTE* original_p1 = p1;
  const BYTE* original_p2 = p2;
  const BYTE* original_mask = mask;

  __m64 v128 = _mm_set1_pi16(0x0080);
  __m64 zero = _mm_setzero_si64();

  int wMod8 = (width/8) * 8;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod8; x += 8) {
      __m64 p1_l = *(reinterpret_cast<const __m64*>(p1+x));
      __m64 p2_l = *(reinterpret_cast<const __m64*>(p2+x));
      __m64 mask_l = *(reinterpret_cast<const __m64*>(mask+x));

      __m64 unpacked_p1_l = _mm_unpacklo_pi8(p1_l, zero);
      __m64 unpacked_p1_h = _mm_unpackhi_pi8(p1_l, zero);

      __m64 unpacked_p2_l = _mm_unpacklo_pi8(p2_l, zero);
      __m64 unpacked_p2_h = _mm_unpackhi_pi8(p2_l, zero);

      __m64 unpacked_mask_l = _mm_unpacklo_pi8(mask_l, zero);
      __m64 unpacked_mask_h = _mm_unpackhi_pi8(mask_l, zero);

      __m64 result_l = overlay_blend_mmx_core(unpacked_p1_l, unpacked_p2_l, unpacked_mask_l, v128);
      __m64 result_h = overlay_blend_mmx_core(unpacked_p1_h, unpacked_p2_h, unpacked_mask_h, v128);

      __m64 result = _m_packuswb(result_l, result_h);

      *reinterpret_cast<__m64*>(p1+x) = result;
    }
    
    // Leftover value
    for (int x = wMod8; x < width; x++) {
      BYTE result = overlay_blend_c_core(p1[x], p2[x], static_cast<int>(mask[x]));
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}
#pragma warning (default: 4799)
#endif
void overlay_blend_sse2_plane_masked(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                     const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                     const int width, const int height) {
        BYTE* original_p1 = p1;
  const BYTE* original_p2 = p2;
  const BYTE* original_mask = mask;

  __m128i v128 = _mm_set1_epi16(0x0080);
  __m128i zero = _mm_setzero_si128();

  int wMod16 = (width/16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      __m128i p1_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1+x));
      __m128i p1_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1+x+8));

      __m128i p2_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p2+x));
      __m128i p2_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p2+x+8));

      __m128i mask_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask+x));
      __m128i mask_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask+x+8));

      __m128i unpacked_p1_l = _mm_unpacklo_epi8(p1_l, zero);
      __m128i unpacked_p1_h = _mm_unpacklo_epi8(p1_h, zero);

      __m128i unpacked_p2_l = _mm_unpacklo_epi8(p2_l, zero);
      __m128i unpacked_p2_h = _mm_unpacklo_epi8(p2_h, zero);

      __m128i unpacked_mask_l = _mm_unpacklo_epi8(mask_l, zero);
      __m128i unpacked_mask_h = _mm_unpacklo_epi8(mask_h, zero);

      __m128i result_l = overlay_blend_sse2_core(unpacked_p1_l, unpacked_p2_l, unpacked_mask_l, v128);
      __m128i result_h = overlay_blend_sse2_core(unpacked_p1_h, unpacked_p2_h, unpacked_mask_h, v128);

      __m128i result = _mm_packus_epi16(result_l, result_h);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1+x), result);
    }
    
    // Leftover value
    for (int x = wMod16; x < width; x++) {
      BYTE result = overlay_blend_c_core(p1[x], p2[x], static_cast<int>(mask[x]));
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}

void overlay_blend_c_plane_opacity(BYTE *p1, const BYTE *p2,
                                   const int p1_pitch, const int p2_pitch,
                                   const int width, const int height, const int opacity) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      BYTE result = overlay_blend_c_core(p1[x], p2[x], opacity);
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
  }
}
#ifdef X86_32
#pragma warning (disable: 4799)
void overlay_blend_mmx_plane_opacity(BYTE *p1, const BYTE *p2,
                                     const int p1_pitch, const int p2_pitch,
                                     const int width, const int height, const int opacity) {
        BYTE* original_p1 = p1;
  const BYTE* original_p2 = p2;

  __m64 v128 = _mm_set1_pi16(0x0080);
  __m64 zero = _mm_setzero_si64();
  __m64 mask = _mm_set1_pi16(static_cast<short>(opacity));

  int wMod8 = (width/8) * 8;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod8; x += 8) {
      __m64 p1_l = *(reinterpret_cast<const __m64*>(p1+x));
      __m64 p2_l = *(reinterpret_cast<const __m64*>(p2+x));

      __m64 unpacked_p1_l = _mm_unpacklo_pi8(p1_l, zero);
      __m64 unpacked_p1_h = _mm_unpackhi_pi8(p1_l, zero);

      __m64 unpacked_p2_l = _mm_unpacklo_pi8(p2_l, zero);
      __m64 unpacked_p2_h = _mm_unpackhi_pi8(p2_l, zero);

      __m64 result_l = overlay_blend_mmx_core(unpacked_p1_l, unpacked_p2_l, mask, v128);
      __m64 result_h = overlay_blend_mmx_core(unpacked_p1_h, unpacked_p2_h, mask, v128);

      __m64 result = _m_packuswb(result_l, result_h);

      *reinterpret_cast<__m64*>(p1+x) = result;
    }

    // Leftover value
    for (int x = wMod8; x < width; x++) {
      BYTE result = overlay_blend_c_core(p1[x], p2[x], opacity);
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
  }
}
#pragma warning (default: 4799)
#endif
void overlay_blend_sse2_plane_opacity(BYTE *p1, const BYTE *p2,
                                      const int p1_pitch, const int p2_pitch,
                                      const int width, const int height, const int opacity) {
        BYTE* original_p1 = p1;
  const BYTE* original_p2 = p2;

  __m128i v128 = _mm_set1_epi16(0x0080);
  __m128i zero = _mm_setzero_si128();
  __m128i mask = _mm_set1_epi16(static_cast<short>(opacity));

  int wMod16 = (width/16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      __m128i p1_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1+x));
      __m128i p1_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1+x+8));

      __m128i p2_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p2+x));
      __m128i p2_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p2+x+8));

      __m128i unpacked_p1_l = _mm_unpacklo_epi8(p1_l, zero);
      __m128i unpacked_p1_h = _mm_unpacklo_epi8(p1_h, zero);

      __m128i unpacked_p2_l = _mm_unpacklo_epi8(p2_l, zero);
      __m128i unpacked_p2_h = _mm_unpacklo_epi8(p2_h, zero);

      __m128i result_l = overlay_blend_sse2_core(unpacked_p1_l, unpacked_p2_l, mask, v128);
      __m128i result_h = overlay_blend_sse2_core(unpacked_p1_h, unpacked_p2_h, mask, v128);

      __m128i result = _mm_packus_epi16(result_l, result_h);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1+x), result);
    }

    // Leftover value
    for (int x = wMod16; x < width; x++) {
      BYTE result = overlay_blend_c_core(p1[x], p2[x], opacity);
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
  }
}

void overlay_blend_c_plane_masked_opacity(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                  const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                  const int width, const int height, const int opacity) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int new_mask = overley_merge_mask_c(mask[x], opacity);
      BYTE result = overlay_blend_c_core(p1[x], p2[x], static_cast<int>(new_mask));
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}
#ifdef X86_32
#pragma warning (disable: 4799)
void overlay_blend_mmx_plane_masked_opacity(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                    const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                    const int width, const int height, const int opacity) {
        BYTE* original_p1 = p1;
  const BYTE* original_p2 = p2;
  const BYTE* original_mask = mask;

  __m64 v128 = _mm_set1_pi16(0x0080);
  __m64 zero = _mm_setzero_si64();
  __m64 opacity_mask = _mm_set1_pi16(static_cast<short>(opacity));

  int wMod8 = (width/8) * 8;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod8; x += 8) {
      __m64 p1_l = *(reinterpret_cast<const __m64*>(p1+x));
      __m64 p2_l = *(reinterpret_cast<const __m64*>(p2+x));
      __m64 mask_l = *(reinterpret_cast<const __m64*>(mask+x));

      __m64 unpacked_p1_l = _mm_unpacklo_pi8(p1_l, zero);
      __m64 unpacked_p1_h = _mm_unpackhi_pi8(p1_l, zero);

      __m64 unpacked_p2_l = _mm_unpacklo_pi8(p2_l, zero);
      __m64 unpacked_p2_h = _mm_unpackhi_pi8(p2_l, zero);

      __m64 unpacked_mask_l = _mm_unpacklo_pi8(mask_l, zero);
      __m64 unpacked_mask_h = _mm_unpackhi_pi8(mask_l, zero);

      unpacked_mask_l = overlay_merge_mask_mmx(unpacked_mask_l, opacity_mask);
      unpacked_mask_h = overlay_merge_mask_mmx(unpacked_mask_h, opacity_mask);

      __m64 result_l = overlay_blend_mmx_core(unpacked_p1_l, unpacked_p2_l, unpacked_mask_l, v128);
      __m64 result_h = overlay_blend_mmx_core(unpacked_p1_h, unpacked_p2_h, unpacked_mask_h, v128);

      __m64 result = _m_packuswb(result_l, result_h);

      *reinterpret_cast<__m64*>(p1+x) = result;
    }

    // Leftover value
    for (int x = wMod8; x < width; x++) {
      int new_mask = overley_merge_mask_c(mask[x], opacity);
      BYTE result = overlay_blend_c_core(p1[x], p2[x], static_cast<int>(new_mask));
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}
#pragma warning (default: 4799)
#endif
void overlay_blend_sse2_plane_masked_opacity(BYTE *p1, const BYTE *p2, const BYTE *mask,
                                     const int p1_pitch, const int p2_pitch, const int mask_pitch,
                                     const int width, const int height, const int opacity) {
        BYTE* original_p1 = p1;
  const BYTE* original_p2 = p2;
  const BYTE* original_mask = mask;

  __m128i v128 = _mm_set1_epi16(0x0080);
  __m128i zero = _mm_setzero_si128();
  __m128i opacity_mask = _mm_set1_epi16(static_cast<short>(opacity));

  int wMod16 = (width/16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      __m128i p1_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1+x));
      __m128i p1_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1+x+8));

      __m128i p2_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p2+x));
      __m128i p2_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p2+x+8));

      __m128i mask_l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask+x));
      __m128i mask_h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask+x+8));

      __m128i unpacked_p1_l = _mm_unpacklo_epi8(p1_l, zero);
      __m128i unpacked_p1_h = _mm_unpacklo_epi8(p1_h, zero);

      __m128i unpacked_p2_l = _mm_unpacklo_epi8(p2_l, zero);
      __m128i unpacked_p2_h = _mm_unpacklo_epi8(p2_h, zero);

      __m128i unpacked_mask_l = _mm_unpacklo_epi8(mask_l, zero);
      __m128i unpacked_mask_h = _mm_unpacklo_epi8(mask_h, zero);

      unpacked_mask_l = overlay_merge_mask_sse2(unpacked_mask_l, opacity_mask);
      unpacked_mask_h = overlay_merge_mask_sse2(unpacked_mask_h, opacity_mask);

      __m128i result_l = overlay_blend_sse2_core(unpacked_p1_l, unpacked_p2_l, unpacked_mask_l, v128);
      __m128i result_h = overlay_blend_sse2_core(unpacked_p1_h, unpacked_p2_h, unpacked_mask_h, v128);

      __m128i result = _mm_packus_epi16(result_l, result_h);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1+x), result);
    }

    // Leftover value
    for (int x = wMod16; x < width; x++) {
      int new_mask = overley_merge_mask_c(mask[x], opacity);
      BYTE result = overlay_blend_c_core(p1[x], p2[x], static_cast<int>(new_mask));
      p1[x] = result;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}

/////////////////////////////////////////////
// Mode: Darken/Lighten
/////////////////////////////////////////////

typedef __m128i (OverlaySseBlendOpaque)(const __m128i&, const __m128i&, const __m128i&);
typedef __m128i (OverlaySseCompare)(const __m128i&, const __m128i&, const __m128i&);
#ifdef X86_32
typedef   __m64 (OverlayMmxCompare)(const __m64&, const __m64&, const __m64&);
#endif
typedef     int (OverlayCCompare)(BYTE, BYTE);

template<OverlayCCompare compare>
void OV_FORCEINLINE overlay_darklighten_c(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int mask = compare(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core(p1V[x], p2V[x], mask);
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }
}
#ifdef X86_32
template<OverlayMmxCompare compare, OverlayCCompare compare_c>
void OV_FORCEINLINE overlay_darklighten_mmx(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  __m64 zero = _mm_setzero_si64();

  int wMod8 = (width/8) * 8;
  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod8; x+=8) {
      // Load Y Plane
      __m64 p1_y = *(reinterpret_cast<const __m64*>(p1Y+x));
      __m64 p2_y = *(reinterpret_cast<const __m64*>(p2Y+x));

      // Compare
      __m64 cmp_result = compare(p1_y, p2_y, zero);

      // Process U Plane
      __m64 result_y = overlay_blend_opaque_mmx_core(p1_y, p2_y, cmp_result);
      *reinterpret_cast<__m64*>(p1Y+x) = result_y;

      // Process U plane
      __m64 p1_u = *(reinterpret_cast<const __m64*>(p1U+x));
      __m64 p2_u = *(reinterpret_cast<const __m64*>(p2U+x));
      
      __m64 result_u = overlay_blend_opaque_mmx_core(p1_u, p2_u, cmp_result);
      *reinterpret_cast<__m64*>(p1U+x) = result_u;

      // Process V plane
      __m64 p1_v = *(reinterpret_cast<const __m64*>(p1V+x));
      __m64 p2_v = *(reinterpret_cast<const __m64*>(p2V+x));
      
      __m64 result_v = overlay_blend_opaque_mmx_core(p1_v, p2_v, cmp_result);
      *reinterpret_cast<__m64*>(p1V+x) = result_v;
    }

    // Leftover value
    for (int x = wMod8; x < width; x++) {
      int mask = compare_c(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core(p1V[x], p2V[x], mask);
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }

  _mm_empty();
}
#endif
template <OverlaySseBlendOpaque blend, OverlaySseCompare compare, OverlayCCompare compare_c>
void OV_FORCEINLINE overlay_darklighten_sse(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  __m128i zero = _mm_setzero_si128();

  int wMod16 = (width/16) * 16;
  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x+=16) {
      // Load Y Plane
      __m128i p1_y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1Y+x));
      __m128i p2_y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2Y+x));

      // Compare
      __m128i cmp_result = compare(p1_y, p2_y, zero);

      // Process U Plane
      __m128i result_y = blend(p1_y, p2_y, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1Y+x), result_y);

      // Process U plane
      __m128i p1_u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1U+x));
      __m128i p2_u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2U+x));
      
      __m128i result_u = blend(p1_u, p2_u, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1U+x), result_u);

      // Process V plane
      __m128i p1_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1V+x));
      __m128i p2_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2V+x));
      
      __m128i result_v = blend(p1_v, p2_v, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1V+x), result_v);
    }

    // Leftover value
    for (int x = wMod16; x < width; x++) {
      int mask = compare_c(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core(p1V[x], p2V[x], mask);
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }
}

// Compare functions for lighten and darken mode
int OV_FORCEINLINE overlay_darken_c_cmp(BYTE p1, BYTE p2) {
  return p2 <= p1;
}
#ifdef X86_32
__m64 OV_FORCEINLINE overlay_darken_mmx_cmp(const __m64& p1, const __m64& p2, const __m64& zero) {
  __m64 diff = _mm_subs_pu8(p2, p1);
  return _mm_cmpeq_pi8(diff, zero);
}
#endif
__m128i OV_FORCEINLINE overlay_darken_sse_cmp(const __m128i& p1, const __m128i& p2, const __m128i& zero) {
  __m128i diff = _mm_subs_epu8(p2, p1);
  return _mm_cmpeq_epi8(diff, zero);
}

int OV_FORCEINLINE overlay_lighten_c_cmp(BYTE p1, BYTE p2) {
  return p2 >= p1;
}
#ifdef X86_32
__m64 OV_FORCEINLINE overlay_lighten_mmx_cmp(const __m64& p1, const __m64& p2, const __m64& zero) {
  __m64 diff = _mm_subs_pu8(p1, p2);
  return _mm_cmpeq_pi8(diff, zero);
}
#endif
__m128i OV_FORCEINLINE overlay_lighten_sse_cmp(const __m128i& p1, const __m128i& p2, const __m128i& zero) {
  __m128i diff = _mm_subs_epu8(p1, p2);
  return _mm_cmpeq_epi8(diff, zero);
}

// Exported function
void overlay_darken_c(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_c<overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_c(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_c<overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}

#ifdef X86_32
void overlay_darken_mmx(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_mmx<overlay_darken_mmx_cmp, overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_mmx(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_mmx<overlay_lighten_mmx_cmp, overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
#endif

void overlay_darken_sse2(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse<overlay_blend_opaque_sse2_core, overlay_darken_sse_cmp, overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_sse2(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse<overlay_blend_opaque_sse2_core, overlay_lighten_sse_cmp, overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}

void overlay_darken_sse41(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse<overlay_blend_opaque_sse41_core, overlay_darken_sse_cmp, overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_sse41(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse<overlay_blend_opaque_sse41_core, overlay_lighten_sse_cmp, overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
