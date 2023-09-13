## A Real-Time Lyrics Alignment System Using Chroma and Phonetic Features for Classical Singing Performance

## Abstract

The goal of real-time lyrics alignment is to take live singing audio as input and to pinpoint the exact position within given lyrics on the fly. 
The task can benefit real-world applications such as the automatic subtitling of live concerts or operas. 
However, designing a real-time model poses a great challenge due to the constraints of only using past input and operating within a minimal latency. 
Furthermore, due to the lack of datasets for real-time models, previous studies have mostly evaluated with private in-house datasets, resulting in a lack of standard evaluation methods. 
This paper presents a real-time lyrics alignment system for classical vocal performances with two contributions. 
First, we improve the lyrics alignment algorithm by finding an optimal combination of chromagram and phonetic posteriorgram (PPG) that capture melodic and phonetics features of the singing voice, respectively. 
Second, we recast the Schubert Winterreise Dataset (SWD) which contains multiple performance renditions of the same pieces as an evaluation set for the real-time lyrics alignment.

## System Design

![](img/overview.jpg)

**Figure 1.** An overview of the proposed real-time lyrics alignment system.

### Offline Phase

<div style="text-align: center;">
    <img src="./img/offline.jpg"  width="60%">
</div>

**Figure 2.** The pipeline of an offline phase aligning *reference* audio and lyrics by extracting *score* audio and annotation from the symbolic score.

The offline phase aims to obtain precise alignment between the *reference* audio and lyrics so that the system can calculate the lyrical position only from *reference* and target audio in online phase, as illustrated in Figure 2. The motivation for the design was to allow the system to automatically generate pseudo-labels from the symbolic score, even if there are no time-aligned lyrics labels for the *reference* audio. Specifically, we extract each onset timings of vocal notes in beat position, syllables of lyrics mapped to specific notes, and the syllabic type (start, end, or middle) from the symbolic score as MusicXML format.

### Online Phase

<div style="text-align: center;">
    <img src="./img/online.png"  width="60%">
</div>

The online phase includes online alignment algorithm between target audio and ref audio with feature extraction on the fly. As Online Dynamic Time Warping (OLTW) achieves linear time complexity and optimizes for real-time by incremental solution, we reproduced the algorithm based on [9] with different configurations suitable for the singing model. We set a sample rate of 16kHz, frame rate of 25, and OLTW window size [15] of 3 seconds. To process audio in real-time, we also implemented a stream processor that enqueues the streaming audio in chunks and extracts features of the target audio. The size of the audio buffer corresponds to 160ms. While the OLTW algorithm is running, the ref position is transferred into the score position via linear interpolation, followed by calculating the score’s beat position.

### Acoustic Model

<div style="text-align: center;">
    <img src="./img/classifier.png"  width="60%">
</div>

**Figure 3.** The network architecture of the proposed acoustic model

Using the CRNN as the backbone architecture, our proposed acoustic model consists of a single CRNN network with a dense layer that takes log-scaled mel-spectrogram and outputs a phonetic posteriorgram (PPG) matrix as illustrated in Fig.3. The overall architecture is taken from the framewise phoneme classifier in [17], but modified to be suitable for real-time. The ConvNet architec- ture, proposed in [18], was used in the CNN part followed by an uni-directional LSTM layer with 1024 size as the RNN part. The last fully-connected layer outputs a PPG matrix with a target size of $N_{phone}$ × $N_{frame}$.

## Datasets

We reconstructed Schubert Winter- reise Dataset(SWD) [23] into winterreise rt, a subset that enables the benchmark evaluation of real-time lyrics alignment models. The SWD dataset is a collection of resources of Schubert’s song cycle for voice and piano ‘Winterreise’, one of the most-performed pieces within the classical music repertoire. 
For detailed information about the dataset, please refer to [here](https://github.com/laurenceyoon/winterreise_rt).

## Evaluation

![](img/table.png)

**Table 1.** Results of offline & online alignment on winterreise rt dataset. Offline alignment results are evaluated with each ground-truth(GT) of score and ref, and online alignment results are evaluated with each GT of ref and target using each feature type (all values are averaged over 24 songs with voice note-level evaluation).

<div style="text-align: center;">
    <img src="./img/discussion.jpg"  width="60%">
</div>

**Figure 4.** TWarping path results with and without phoneme features of ‘No. 11, Fru ̈hlingstraum’ from Winterreise song cycle. The red dots represent GT pairs for voice notes, the purple dots with the warping path results of each model, and the white line with the silence part.

## Demo Video

<div class="youtube-wrapper">
  <iframe width="832" height="468" src="https://www.youtube.com/embed/FNcr7OByZ94?si=aQoZqSp24QypxbEd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>
