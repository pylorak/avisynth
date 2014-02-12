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


#include "greyscale.h"
#include "../core/internal.h"
#include <emmintrin.h>
#include <avs/alignment.h>
#include <avs/minmax.h>
#include <avs/win.h>


/*************************************
 *******   Convert to Greyscale ******
 ************************************/

extern const AVSFunction Greyscale_filters[] = {
  { "Greyscale", "c[matrix]s", Greyscale::Create },       // matrix can be "rec601", "rec709" or "Average"
  { "Grayscale", "c[matrix]s", Greyscale::Create },
  { 0 }
};

Greyscale::Greyscale(PClip _child, const char* matrix, IScriptEnvironment* env)
 : GenericVideoFilter(_child)
{
  matrix_ = Rec601;
  if (matrix) {
    if (!vi.IsRGB())
      env->ThrowError("GreyScale: invalid \"matrix\" parameter (RGB data only)");
    if (!lstrcmpi(matrix, "rec709"))
      matrix_ = Rec709;
    else if (!lstrcmpi(matrix, "Average"))
      matrix_ = Average;
    else if (!lstrcmpi(matrix, "rec601"))
      matrix_ = Rec601;
    else
      env->ThrowError("GreyScale: invalid \"matrix\" parameter (must be matrix=\"Rec601\", \"Rec709\" or \"Average\")");
  }
}

//this is not really faster than MMX but a lot cleaner
static void greyscale_yuy2_sse2(BYTE *srcp, size_t /*width*/, size_t height, size_t pitch) {
  __m128i luma_mask = _mm_set1_epi16(0x00FF);
#pragma warning(push)
#pragma warning(disable: 4309)
  __m128i chroma_value = _mm_set1_epi16(0x8000);
#pragma warning(pop)
  BYTE* end_point = srcp + pitch * height;

  while(srcp < end_point) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    src = _mm_and_si128(src, luma_mask);
    src = _mm_or_si128(src, chroma_value);
    _mm_store_si128(reinterpret_cast<__m128i*>(srcp), src);

    srcp += 16;
  }
}

static void greyscale_rgb32_sse2(BYTE *srcp, size_t /*width*/, size_t height, size_t pitch, int cyb, int cyg, int cyr) {
  __m128i matrix = _mm_set_epi16(0, cyr, cyg, cyb, 0, cyr, cyg, cyb);
  __m128i zero = _mm_setzero_si128();
  __m128i round_mask = _mm_set1_epi32(16384);
  __m128i alpha_mask = _mm_set1_epi32(0xFF000000);

  BYTE* end_point = srcp + pitch * height;

  while(srcp < end_point) { 
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    __m128i alpha = _mm_and_si128(src, alpha_mask);
    __m128i pixel01 = _mm_unpacklo_epi8(src, zero);
    __m128i pixel23 = _mm_unpackhi_epi8(src, zero);

    pixel01 = _mm_madd_epi16(pixel01, matrix);
    pixel23 = _mm_madd_epi16(pixel23, matrix);

    __m128i tmp = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel01), _mm_castsi128_ps(pixel23), _MM_SHUFFLE(3, 1, 3, 1))); // r3*cyr | r2*cyr | r1*cyr | r0*cyr
    __m128i tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel01), _mm_castsi128_ps(pixel23), _MM_SHUFFLE(2, 0, 2, 0)));

    tmp = _mm_add_epi32(tmp, tmp2); 
    tmp = _mm_add_epi32(tmp, round_mask); 
    tmp = _mm_srli_epi32(tmp, 15); // 0 0 0 p3 | 0 0 0 p2 | 0 0 0 p1 | 0 0 0 p0

    //todo: pshufb?
    __m128i result = _mm_or_si128(tmp, _mm_slli_si128(tmp, 1)); 
    result = _mm_or_si128(result, _mm_slli_si128(tmp, 2));
    result = _mm_or_si128(alpha, result);

    _mm_store_si128(reinterpret_cast<__m128i*>(srcp), result);

    srcp += 16;
  }
}


PVideoFrame Greyscale::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  if (vi.IsY8())
    return frame;

  env->MakeWritable(&frame);
  BYTE* srcp = frame->GetWritePtr();
  int pitch = frame->GetPitch();
  int height = vi.height;
  int width = vi.width;

  if (vi.IsPlanar()) {
    memset(frame->GetWritePtr(PLANAR_U), 0x80808080, frame->GetHeight(PLANAR_U) * frame->GetPitch(PLANAR_U));
    memset(frame->GetWritePtr(PLANAR_V), 0x80808080, frame->GetHeight(PLANAR_V) * frame->GetPitch(PLANAR_V));
    return frame;
  }

  if (vi.IsYUY2()) 
  {
    if ((env->GetCPUFlags() & CPUF_SSE2) && width > 4 && IsPtrAligned(srcp, 16)) {
      greyscale_yuy2_sse2(srcp, width, height, pitch);
    }
    else
    {
      for (int y = 0; y<height; ++y)
      {
        for (int x = 0; x<width; x++)
          srcp[x*2+1] = 128;
        srcp += pitch;
      }
    }

    return frame;
  }

  if (vi.IsRGB32() && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16)) 
  {
    const int cyav = int(0.33333*32768+0.5);

    const int cyb = int(0.114*32768+0.5);
    const int cyg = int(0.587*32768+0.5);
    const int cyr = int(0.299*32768+0.5);

    const int cyb709 = int(0.0722*32768+0.5);
    const int cyg709 = int(0.7152*32768+0.5);
    const int cyr709 = int(0.2126*32768+0.5);

    switch (matrix_)
    {
    case Rec601:
      greyscale_rgb32_sse2(srcp, width, height, pitch, cyb, cyg, cyr);
      break;
    case Rec709:
      greyscale_rgb32_sse2(srcp, width, height, pitch, cyb709, cyg709, cyr709);
      break;
    case Average:
      greyscale_rgb32_sse2(srcp, width, height, pitch, cyav, cyav, cyav);
      break;
    }
  }

  if (vi.IsRGB())
  {  // RGB C.
    BYTE* p_count = srcp;

    const int rgb_inc = vi.IsRGB32() ? 4 : 3;
    if (matrix_ == Rec709) {
      //	  const int cyb709 = int(0.0722*65536+0.5); //  4732
      //	  const int cyg709 = int(0.7152*65536+0.5); // 46871
      //	  const int cyr709 = int(0.2126*65536+0.5); // 13933

      for (int y = 0; y<vi.height; ++y) {
        for (int x = 0; x<vi.width; x++) {
          int greyscale = ((srcp[0]*4732)+(srcp[1]*46871)+(srcp[2]*13933)+32768)>>16; // This is the correct brigtness calculations (standardized in Rec. 709)
          srcp[0] = srcp[1] = srcp[2] = greyscale;
          srcp += rgb_inc;
        }
        p_count += pitch;
        srcp = p_count;
      }
    } else if (matrix_ == Average) {
      //	  const int cyav = int(0.333333*65536+0.5); //  21845

      for (int y = 0; y<vi.height; ++y) {
        for (int x = 0; x<vi.width; x++) {
          int greyscale = ((srcp[0]+srcp[1]+srcp[2])*21845+32768)>>16; // This is the average of R, G & B
          srcp[0] = srcp[1] = srcp[2] = greyscale;
          srcp += rgb_inc;
        }
        p_count += pitch;
        srcp = p_count;
      }
    } else {
      //	  const int cyb = int(0.114*65536+0.5); //  7471
      //	  const int cyg = int(0.587*65536+0.5); // 38470
      //	  const int cyr = int(0.299*65536+0.5); // 19595

      for (int y = 0; y<vi.height; ++y) {
        for (int x = 0; x<vi.width; x++) {
          int greyscale = ((srcp[0]*7471)+(srcp[1]*38470)+(srcp[2]*19595)+32768)>>16; // This produces similar results as YUY2 (luma calculation)
          srcp[0] = srcp[1] = srcp[2] = greyscale;
          srcp += rgb_inc;
        }
        p_count += pitch;
        srcp = p_count;
      }
    }
  }
  return frame;
}


AVSValue __cdecl Greyscale::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const VideoInfo& vi = clip->GetVideoInfo();

  if (vi.IsY8())
    return clip;

  return new Greyscale(clip, args[1].AsString(0), env);
}
