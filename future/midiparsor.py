from music21 import *
from collections import defaultdict
import sys
import numpy as np
import pickle


def get_midi(midi_file_path):
    midi = converter.parse(midi_file_path)
    return midi

def preview_midi(midi):
    s2 = instrument.partitionByInstrument(midi)
    for idx, part in enumerate(s2):
        print(f"part_id: {idx}, part: {part}")


def get_pitches(midi, part_id=None):
    """Returns a list of pitches from input midi data
    
    Args:
        midi (midi): midi type from get_midi()
    
    Returns:
        pitch: pitches
    """
    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        print(len(s2))
        if part_id is not None:
            notes = s2.parts[part_id].recurse()
            pitches = [c.pitches for c in notes]
            pitch_arr = np.array(pitches)[None]
            ps_arr = np.array([pitch.ps for pitch in pitches])
        else:
            pitch_arr = []
            ps_arr = []
            for part in s2:
                notes = part.recurse()
                pitches = [c.pitches for c in notes]
                pss = [pitch.ps for pitch in pitches]
                pitch_arr.append(pitches)
                ps_arr.append(pss)
            pitch_arr = np.array(pitch_arr)
            ps_arr = np.array(ps_arr)
    except: # file has notes in a flat structure
        notes = midi.flat.notes
        pitch_arr = np.array([c.pitches for c in notes])
        ps_arr = np.array([pitch.ps for pitch in pitches])
    return pitch_arr, ps_arr


def get_dynamics(midi):
    pass


# Returns <filename> as a list of (pitch, relative offset, duration) tuples
def read_file_as_pitch_offset_duration(filename, dechord=True):
    score = midi.translate.midiFilePathToStream(filename, quarterLengthDivisors=(32,))
    events = score.flat
    processed = []
    print("processing ", filename, "...")
    for i in range(len(events)):
        elt = events[i]
        if isinstance(elt, chord.Chord):
            offset = elt.offset
            duration = elt.quarterLength
            if dechord==True:
                processed.append((elt[0].pitch.ps, offset, duration))
            else:
                for n in elt:
                    processed.append((n.pitch.ps, offset, duration))
        if isinstance(elt, note.Rest) or isinstance(elt, note.Note):
            pitch = 0 if isinstance(elt, note.Rest) else elt.pitch.midi
            offset = elt.offset
            duration = elt.quarterLength
            processed.append((pitch, offset, duration))
    processed.sort(key = lambda x: (x[1], x[0]))
    prev_abs_offset = 0
    for i in range(len(processed)):
        curr_abs_offset = processed[i][1]
        processed[i] = (processed[i][0], curr_abs_offset - prev_abs_offset, processed[i][2])
        prev_abs_offset = curr_abs_offset
    return np.array(processed)


def output_pitch_offset_duration_as_midi_file(arr, output_file):
    input_len = arr.shape[0]
    pitches, offsets, durations = arr[:,0], arr[:,1], arr[:,2]
    # pitches = pitches.astype(np.int32).tolist()
    total_offset = 0
    midi_stream = stream.Stream()
    for idx in range(arr.shape[0]):
        if pitches[idx]!=0.:
            tmp_note = note.Note()
            tmp_note.pitch.ps = pitches[idx]
            tmp_note.duration.quarterLength = durations[idx]
            total_offset += offsets[idx]
            tmp_note.offset = total_offset
            
            # if tmp_note.pitch.ps!=44.:
            #     midi_stream.insert(tmp_note)
            midi_stream.insert(tmp_note)
        else:
            total_offset += offsets[idx]
                   
    midi_stream.write('midi', fp=output_file)


if __name__ == "__main__":
    x = read_file_as_pitch_offset_duration(sys.argv[1], dechord=False)
    print(x.shape)
    output_pitch_offset_duration_as_midi_file(x, "test.mid")