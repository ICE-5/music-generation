import pickle
from music21 import chord, converter, instrument, midi, note, meter
import numpy as np
import sys


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
def read_file_as_pitch_offset_duration(filename):
    score = midi.translate.midiFilePathToStream(filename, quarterLengthDivisors=(32,))
    events = score.flat
    processed = []
    print("processing ", filename, "...")
    for i in range(len(events)):  # flat converts relative offsets into absolute offsets!
        elt = events[i]
        if isinstance(elt, chord.Chord):
            offset = elt.offset
            duration = elt.quarterLength
            for n in elt:
                processed.append((n.pitch.midi, offset, duration))
        if isinstance(elt, note.Rest) or isinstance(elt, note.Note):  # for now, ignoring meter.TimeSignature, tempo.MetronomeMark
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
    return processed


def output_pitch_offset_duration_as_midi_file(arr, output_file):
    input_len = arr.shape[0]
    pitch, offset, duration = arr[:,0], arr[:,1], arr[:,2]
    pitch = pitch.astype(np.int32).tolist()  # must convert to native int type (see final case in Chord's _add_core_or_init)
    print(pitch[:10], offset[:10], duration[:10])

    # key idea: for every offset, maintain a dict that maps: duration --> list of pitches
    # each element of the map is a chord (or note, if only 1)
    i = 0
    total_offset = 0  # TODO: need to update this!!!!
    output_notes = []
    while i < input_len:
        total_offset += offset[i]
        dd = defaultdict(list)
        if pitch[i] != 0:
            dd[duration[i]].append(pitch[i])
        while i+1 < input_len and offset[i+1] == 0:  # increment i and add next values to the map
            i += 1
            if pitch[i] != 0:
                dd[duration[i]].append(pitch[i])
        for d, pitches in dd.items():
            print("(d, pitches) = ", d, ", ", pitches)
            # might be able to express notes as single-element chords...
            item = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            item.offset = total_offset
            item.duration.quarterLength = d
            output_notes.append(item)
        i += 1

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)


if __name__ == "__main__":
    x = np.array(read_file_as_pitch_offset_duration(sys.argv[1]))
    print(x.shape)