import io
import librosa
import numpy as np
import streamlit as st
import audiomentations
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
import pydub

plt.rcParams["figure.figsize"] = (10, 7)


def create_pipeline(transformations: list):
    pipeline = []
    for index, transformation in enumerate(transformations):
        if transformation:
            pipeline.append(index_to_transformation(index))

    return pipeline


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile


@st.cache
def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate


def plot_wave(y, sr):
    fig, ax = plt.subplots()

    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

    return plt.gcf()


def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()


def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)


def plot_audio_transformations(y, sr, pipeline: audiomentations.Compose):
    cols = [1, 1, 1]

    col1, col2, col3 = st.columns(cols)
    with col1:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Original</h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Audio</h5>",
            unsafe_allow_html=True,
        )
        spacing()
        st.audio(create_audio_player(y, sr))
    st.markdown("---")

    y = y
    sr = sr
    for col_index, individual_transformation in enumerate(pipeline.transforms):
        transformation_name = (
            str(type(individual_transformation)).split("'")[1].split(".")[-1]
        )
        modified = individual_transformation(y, sr)
        fig = plot_transformation(modified, sr, transformation_name=transformation_name)
        y = modified

        col1, col2, col3 = st.columns(cols)

        with col1:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>{transformation_name}</h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(fig)
        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_wave(modified, sr))
            spacing()

        with col3:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Audio</h5>",
                unsafe_allow_html=True,
            )
            spacing()
            st.audio(create_audio_player(modified, sr))
        st.markdown("---")
        plt.close("all")


def load_audio_sample(file):
    y, sr = librosa.load(file, sr=22050)

    return y, sr


def index_to_transformation(index: int):
    if index == 0:
        return audiomentations.AddGaussianNoise(p=1.0)
    elif index == 1:
        return audiomentations.AddGaussianSNR(p=1.0, min_snr_in_db=30, max_snr_in_db=90)
    elif index == 2:
        return audiomentations.FrequencyMask(p=1.0)
    elif index == 3:
        return audiomentations.TimeMask(p=1.0)
    elif index == 4:
        return audiomentations.TimeStretch(p=1.0)
    elif index == 5:
        return audiomentations.PitchShift(p=1.0)
    elif index == 6:
        return audiomentations.Shift(p=1.0)
    elif index == 7:
        return audiomentations.Normalize(p=1.0)
    elif index == 8:
        return audiomentations.PolarityInversion(p=1.0)
    elif index == 9:
        return audiomentations.Gain(p=1.0)
    elif index == 10:
        return audiomentations.AddBackgroundNoise(sounds_path="background_noise", p=1.0)
    elif index == 11:
        return audiomentations.AddShortNoises(sounds_path="background_noise", p=1.0)
    elif index == 12:
        return audiomentations.ClippingDistortion(max_percentile_threshold=10, p=1.0)
    elif index == 13:
        return audiomentations.Clip(p=1.0)
    elif index == 14:
        return audiomentations.HighPassFilter(p=1.0)
    elif index == 15:
        return audiomentations.LowPassFilter(p=1.0)
    elif index == 16:
        return audiomentations.BandPassFilter(p=1.0)
    elif index == 17:
        return audiomentations.Reverse(p=1.0)
    elif index == 18:
        return audiomentations.BandStopFilter(p=1.0)
    elif index == 19:
        return audiomentations.PeakingFilter(p=1.0)
    elif index == 20:
        return audiomentations.LowShelfFilter(p=1.0)
    elif index == 21:
        return audiomentations.HighShelfFilter(p=1.0)
    elif index == 22:
        return audiomentations.GainTransition(p=1.0)
    elif index == 23:
        return audiomentations.RoomSimulator(p=1.0)
    elif index == 24:
        return audiomentations.Padding(p=1.0)
    elif index == 25:
        return audiomentations.SevenBandParametricEQ(p=1.0)
    elif index == 26:
        return audiomentations.AirAbsorption(p=1.0)
    elif index == 27:
        return audiomentations.Limiter(p=1.0)


def action(file_uploader, selected_provided_file, transformations):
    if file_uploader is not None:
        y, sr = handle_uploaded_audio_file(file_uploader)
    else:
        if selected_provided_file == "Dog":
            y, sr = librosa.load("samples/dog.wav")
        elif selected_provided_file == "Cow":
            y, sr = librosa.load("samples/cow.wav")
        elif selected_provided_file == "Thunder":
            y, sr = librosa.load("samples/thunder.wav")

    pipeline = audiomentations.Compose(create_pipeline(transformations))
    plot_audio_transformations(y, sr, pipeline)


def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown(
        "# Visualize an audio pipeline\n"
        "### Select the components of the pipeline in the sidebar.\n"
        "Once you have chosen augmentation techniques, select or upload an audio file\n. "
        'Then click "Apply" to start! \n\n'
        'For more information see the corresponding [blog post](https://towardsdatascience.com/visualizing-audio-pipelines-with-streamlit-96525781b5d9) and check out [the source code on GitHub](https://github.com/phrasenmaeher/audio-transformation-visualization).'
    )
    placeholder2.markdown(
        "After clicking start, the individual steps of the pipeline are visualized. The ouput of the previous step is the input to the next step."
    )
    # placeholder.write("Create your audio pipeline by selecting augmentations in the sidebar.")
    st.sidebar.markdown("Choose the transformations here:")
    gaussian_noise = st.sidebar.checkbox("GaussianNoise")
    gaussian_noise_snr = st.sidebar.checkbox("GaussianNoise with random SNR")
    frequency_mask = st.sidebar.checkbox("FrequencyMask")
    time_mask = st.sidebar.checkbox("TimeMask")
    time_strech = st.sidebar.checkbox("TimeStretch")
    pitch_shift = st.sidebar.checkbox("PitchShift")
    shift = st.sidebar.checkbox("Shift")
    normalize = st.sidebar.checkbox("(Peak-)Normalize")
    polarity_inversion = st.sidebar.checkbox("PolarityInversion")
    gain = st.sidebar.checkbox("Gain")
    background_noise = st.sidebar.checkbox(
        "AddBackgroundNoise", help="Adds a random background noise"
    )
    add_short_noises = st.sidebar.checkbox(
        "AddShortNoises", help="Mixes bursts of random sounds into the audio signal"
    )
    clipping_distortion = st.sidebar.checkbox("ClippingDistortion")
    clip = st.sidebar.checkbox("Clip")
    highpass = st.sidebar.checkbox("HighPassFilter")
    lowpass = st.sidebar.checkbox("LowPassFilter")
    bandpass = st.sidebar.checkbox("BandPassFilter")
    reverse = st.sidebar.checkbox("Reverse")
    bandstop = st.sidebar.checkbox("BandStopFilter")
    peaking = st.sidebar.checkbox("PeakingFilter")
    lowshelf = st.sidebar.checkbox("LowShelfFilter")
    highshelf = st.sidebar.checkbox("HighShelfFilter")
    gain_transition = st.sidebar.checkbox("GainTransition")
    room_simulator = st.sidebar.checkbox("RoomSimulator")
    padding = st.sidebar.checkbox("Padding")
    seven_band_eq = st.sidebar.checkbox("SevenBandParametricEQ")
    air_absorption = st.sidebar.checkbox("AirAbsorption")
    limiter = st.sidebar.checkbox("Limiter")

    st.sidebar.markdown("---")
    st.sidebar.markdown("(Optional) Upload an audio file here:")
    file_uploader = st.sidebar.file_uploader(
        label="", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"]
    )
    st.sidebar.markdown("Or select a sample file here:")
    selected_provided_file = st.sidebar.selectbox(
        label="", options=["Cow", "Dog", "Thunder"]
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Apply"):
        placeholder.empty()
        placeholder2.empty()
        transformations = [
            gaussian_noise,
            gaussian_noise_snr,
            frequency_mask,
            time_mask,
            time_strech,
            pitch_shift,
            shift,
            normalize,
            polarity_inversion,
            gain,
            background_noise,
            add_short_noises,
            clipping_distortion,
            clip,
            highpass,
            lowpass,
            bandpass,
            reverse,
            bandstop,
            peaking,
            lowshelf,
            highshelf,
            gain_transition,
            room_simulator,
            padding,
            seven_band_eq,
            air_absorption,
            limiter,

        ]

        action(
            file_uploader=file_uploader,
            selected_provided_file=selected_provided_file,
            transformations=transformations,
        )


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Audio augmentation visualization")
    main()
