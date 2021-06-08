import io
import librosa
import numpy as np
import streamlit as st
import audiomentations
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
import wave
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


def handle_uploaded_audio_file(uploaded_file):
    with wave.open(uploaded_file, mode="rb") as f:
        wave_object = f

    params = wave_object.getparams()

    a = pydub.AudioSegment.from_raw(file=uploaded_file, sample_width=params.sampwidth,
                                    frame_rate=params.framerate, channels=params.nchannels)

    samples = a.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max

    return fp_arr, params.framerate


def plot_wave(y, sr):
    fig, ax = plt.subplots()
    img = librosa.display.waveplot(y, sr=sr, x_axis='time', ax=ax)

    return plt.gcf()


def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    # ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()


def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)


def plot_audio_transformations(y, sr, pipeline: audiomentations.Compose):
    cols = [1, 1, 1]

    col1, col2, col3 = st.beta_columns(cols)
    with col1:
        st.markdown(f"<h4 style='text-align: center; color: black;'>Original</h5>",
                    unsafe_allow_html=True)
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                    unsafe_allow_html=True)
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(f"<h4 style='text-align: center; color: black;'>Audio</h5>",
                    unsafe_allow_html=True)
        spacing()
        st.audio(create_audio_player(y, sr))
    st.markdown("---")

    y = y
    sr = sr
    for col_index, individual_transformation in enumerate(pipeline.transforms):
        transformation_name = str(type(individual_transformation)).split("'")[1].split(".")[-1]
        modified = individual_transformation(y, sr)
        fig = plot_transformation(modified, sr, transformation_name=transformation_name)
        y = modified

        col1, col2, col3 = st.beta_columns(cols)

        with col1:
            st.markdown(f"<h4 style='text-align: center; color: black;'>{transformation_name}</h5>",
                        unsafe_allow_html=True)
            st.pyplot(fig)
        with col2:
            st.markdown(f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                        unsafe_allow_html=True)
            st.pyplot(plot_wave(modified, sr))
            spacing()

        with col3:
            st.markdown(f"<h4 style='text-align: center; color: black;'>Audio</h5>",
                        unsafe_allow_html=True)
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
        return audiomentations.AddGaussianSNR(p=1.0)
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


def main():
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
    st.sidebar.markdown("---")
    st.sidebar.markdown("Upload an audio file:")
    file_uploader = st.sidebar.file_uploader(label="", type=".wav")

    transformations = [gaussian_noise, gaussian_noise_snr, frequency_mask, time_mask, time_strech, pitch_shift, shift,
                       normalize, polarity_inversion, gain]

    st.sidebar.markdown("---")
    if st.sidebar.button("Apply"):
        if file_uploader is not None:
            y, sr = handle_uploaded_audio_file(file_uploader)

            pipeline = audiomentations.Compose(create_pipeline(transformations))
            plot_audio_transformations(y, sr, pipeline)


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Audio visualization")
    main()
