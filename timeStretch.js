function createFadeBuffer(context, activeTime, fadeTime) {
  const sr = 48000;

  const length1 = Math.round(activeTime * sr);
  const length2 = Math.round((activeTime - 2 * fadeTime) * sr);
  const length = length1 + length2;

  const fadeLength = Math.round(fadeTime * sr);

  const fadeStart = fadeLength;
  const fadeEnd = length1 - fadeLength;

  const work = new Float64Array(length);

  const halfPi = Math.PI * 0.5;

  for (let i = 0; i < length1; i++) {
    if (i < fadeStart) {
      const t = i / fadeLength;
      work[i] = Math.sin(halfPi * t);
    } else if (i >= fadeEnd) {
      const t = (i - fadeEnd) / fadeLength;
      work[i] = Math.cos(halfPi * t);
    } else {
      work[i] = 1.0;
    }
  }

  const buffer = context.createBuffer(1, length, sr);
  const out = buffer.getChannelData(0);

  for (let i = 0; i < length; i++) {
    out[i] = work[i];
  }

  return buffer;
}

function createDelayTimeBuffer(context, activeTime, fadeTime, shiftUp, semitoneShift = 0) {
  const sampleRate = 48000;
  const length1 = Math.round(activeTime * sampleRate);
  const length2 = Math.round((activeTime - 2 * fadeTime) * sampleRate);
  const length = length1 + length2;

  const buffer = context.createBuffer(2, length, sampleRate);
  const p0 = buffer.getChannelData(0);
  const p1 = buffer.getChannelData(1);

  const fftSize = 2048;
  const hopSize = 512;
  const numBins = fftSize / 2 + 1;
  
  const stretchFactor = shiftUp 
    ? Math.pow(2, semitoneShift / 12) 
    : Math.pow(2, -semitoneShift / 12);
  
  const outputLength = Math.round(length1 / stretchFactor);
  
  const real = new Float64Array(numBins);
  const imag = new Float64Array(numBins);
  const window = new Float64Array(fftSize);
  const lastPhase = new Float64Array(numBins);
  const sumPhase = new Float64Array(numBins);
  
  for (let i = 0; i < fftSize; i++) {
    window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (fftSize - 1));
  }
  
  const input = new Float64Array(length1);
  for (let i = 0; i < length1; i++) {
    const fadeStart = Math.round(fadeTime * sampleRate);
    const fadeEnd = length1 - fadeStart;
    
    if (i < fadeStart) {
      const t = i / fadeStart;
      input[i] = Math.sin(Math.PI * 0.5 * t);
    } else if (i >= fadeEnd) {
      const t = (i - fadeEnd) / fadeStart;
      input[i] = Math.cos(Math.PI * 0.5 * t);
    } else {
      input[i] = 1.0;
    }
  }
  
  const output = new Float64Array(outputLength);
  let outputIndex = 0;
  let inputIndex = 0;
  let frameCount = 0;
  
  const omega = new Float64Array(numBins);
  for (let k = 0; k < numBins; k++) {
    omega[k] = 2 * Math.PI * k * hopSize / fftSize;
  }
  
  while (inputIndex <= length1 - fftSize && outputIndex < outputLength - hopSize) {
    for (let i = 0; i < fftSize; i++) {
      real[i] = input[inputIndex + i] * window[i];
      imag[i] = 0;
    }
    
    fft(real, imag);
    
    for (let k = 0; k < numBins; k++) {
      const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
      const phase = Math.atan2(imag[k], real[k]);
      
      const deltaPhase = phase - lastPhase[k] - omega[k];
      let wrappedDelta = deltaPhase;
      while (wrappedDelta > Math.PI) wrappedDelta -= 2 * Math.PI;
      while (wrappedDelta < -Math.PI) wrappedDelta += 2 * Math.PI;
      
      sumPhase[k] += omega[k] + wrappedDelta;
      lastPhase[k] = phase;
      
      const newMag = mag;
      const newPhase = sumPhase[k];
      
      real[k] = newMag * Math.cos(newPhase);
      imag[k] = newMag * Math.sin(newPhase);
    }
    
    ifft(real, imag);
    
    for (let i = 0; i < fftSize; i++) {
      if (outputIndex + i < outputLength) {
        output[outputIndex + i] += real[i] * window[i] * 0.5;
      }
    }
    
    inputIndex += hopSize;
    outputIndex += Math.round(hopSize * stretchFactor);
    frameCount++;
  }
  
  for (let i = 0; i < outputLength; i++) {
    const pos = (i / outputLength) * length1;
    const fadeStart = Math.round(fadeTime * sampleRate);
    const fadeEnd = length1 - fadeStart;
    
    let gain = 1.0;
    if (pos < fadeStart) {
      gain = Math.sin(Math.PI * 0.5 * (pos / fadeStart));
    } else if (pos >= fadeEnd) {
      gain = Math.cos(Math.PI * 0.5 * ((pos - fadeEnd) / fadeStart));
    }
    
    p0[i] = output[i] * gain;
    p1[i] = output[i] * gain;
  }
  
  for (let i = length1; i < length; i++) {
    p0[i] = 0;
    p1[i] = 0;
  }

  return buffer;

  function fft(re, im) {
    const n = re.length;
    const bits = Math.log2(n);
    
    for (let i = 0; i < n; i++) {
      const j = bitReverse(i, bits);
      if (j > i) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    
    for (let len = 2; len <= n; len *= 2) {
      const half = len / 2;
      const angle = -2 * Math.PI / len;
      
      for (let i = 0; i < n; i += len) {
        for (let j = 0; j < half; j++) {
          const tRe = Math.cos(angle * j) * re[i + j + half] - Math.sin(angle * j) * im[i + j + half];
          const tIm = Math.sin(angle * j) * re[i + j + half] + Math.cos(angle * j) * im[i + j + half];
          
          re[i + j + half] = re[i + j] - tRe;
          im[i + j + half] = im[i + j] - tIm;
          re[i + j] += tRe;
          im[i + j] += tIm;
        }
      }
    }
  }
  
  function ifft(re, im) {
    const n = re.length;
    
    for (let i = 1; i < n / 2; i++) {
      [re[i], re[n - i]] = [re[n - i], re[i]];
      [im[i], im[n - i]] = [im[n - i], im[i]];
    }
    
    fft(re, im);
    
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
  
  function bitReverse(i, bits) {
    let result = 0;
    for (let b = 0; b < bits; b++) {
      result = (result << 1) | (i & 1);
      i >>= 1;
    }
    return result;
  }
}