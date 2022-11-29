import os
from dataclasses import astuple, dataclass
from enum import Enum
from fractions import Fraction
from math import gcd
from random import choice, randint, random
from typing import Generic, Literal, NamedTuple, TypeVar
import argparse

import mido
import music21

from errors import NoSetTempoMessageFound, NoTimeSignatureMessageFound

# ------------------------ Composition parsing --------------------------------


class Time_signature(NamedTuple):
    """A class representing a time signature."""

    numerator: int
    denominator: int

    def __str__(self):
        return f'{self.numerator}/{self.denominator}'

    def __repr__(self):
        return self.__str__()


class Duration(NamedTuple):
    """A class representing a musical duration"""

    numerator: int
    denominator: int

    def __str__(self):
        return f'{self.numerator}/{self.denominator}'

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return Duration(
            numerator=self.numerator * other.denominator + other.numerator
            * self.denominator,
            denominator=self.denominator * other.denominator
        )

    def __truediv__(self, other) -> Fraction:
        return Fraction(
            numerator=self.numerator * other.denominator,
            denominator=self.denominator * other.numerator
        )

    def __mul__(self, other: int):
        return Duration(
            numerator=self.numerator * other,
            denominator=self.denominator
        )

    def __eq__(self, other):
        first = self.numerator * other.denominator
        second = self.denominator * other.numerator
        return first == second

    def __ge__(self, other):
        first = self.numerator * other.denominator
        second = self.denominator * other.numerator
        return first >= second

    def __gt__(self, other):
        first = self.numerator * other.denominator
        second = self.denominator * other.numerator
        return first > second

    def __le__(self, other):
        first = self.numerator * other.denominator
        second = self.denominator * other.numerator
        return first <= second

    def __lt__(self, other):
        first = self.numerator * other.denominator
        second = self.denominator * other.numerator
        return first < second

    def __ne__(self, other):
        first = self.numerator * other.denominator
        second = self.denominator * other.numerator
        return first != second


def duration_to_clocks(
    duration: Duration, clocks_per_beat: int,
) -> int:
    """Convert a Duration to the number of clocks"""
    return int(
        duration.numerator
        * clocks_per_beat
        * 4
        / duration.denominator
    )


def duration_from_fraction(
    fraction: Fraction, time_signature: Time_signature
) -> Duration:
    """Convert a Fraction to a Duration"""
    if fraction.denominator % time_signature.denominator == 0:
        return Duration(fraction.numerator, fraction.denominator)
    else:
        greatest_common_divisor = gcd(
            fraction.denominator, time_signature.denominator
        )
        return Duration(
            int(
                fraction.numerator
                * time_signature.denominator
                / greatest_common_divisor
            ),
            int(
                fraction.denominator
                * time_signature.denominator
                / greatest_common_divisor
            ),
        )


BPM = float
"""Beats per minute"""

Pitch = int
"""Pitch of a note"""


class Key(Enum):
    """A class representing a key"""

    C = 0
    C_SHARP = 1
    D = 2
    D_SHARP = 3
    E = 4
    F = 5
    F_SHARP = 6
    G = 7
    G_SHARP = 8
    A = 9
    A_SHARP = 10
    B = 11

    def __str__(self):
        return self.name.replace('_SHARP', '#')

    def __repr__(self):
        return self.__str__()


class Note(NamedTuple):
    """A note with a pitch and a duration"""

    pitch: Pitch
    duration: Duration

    def get_key(self) -> Key:
        return Key(self.pitch % 12)

    def get_octave(self) -> int:
        return self.pitch // 12 - 2

    def __str__(self):
        octave = self.get_octave()
        key = self.get_key()
        return f'{key}{octave}, {self.duration}'

    def __repr__(self):
        return self.__str__()


class Chord_in_keys(NamedTuple):
    """A class representing a chord"""

    notes: list[Key]
    duration: Duration

    def __str__(self):
        return f'{self.notes}'

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return Chord_in_keys(self.notes.copy(), self.duration)

    def to_pitches(self, octave: int) -> list[Pitch]:
        return [note.value + 12 * octave for note in self.notes]


class Chord_function(Enum):
    TONIC = 0
    SUBDOMINANT = 1
    DOMINANT = 2


T = TypeVar('T')


@dataclass
class Event_in_time(Generic[T]):
    event: T
    start: Duration

    def __iter__(self):
        yield from astuple(self)

    def __str__(self):
        return f'{self.start}: {self.event}'

    def __repr__(self):
        return self.__str__()


class Timeline(NamedTuple):
    """A class representing a timeline"""

    Start = Duration
    melody: list[Event_in_time[Note]]
    chords: list[Event_in_time[Chord_in_keys]]

    def __str__(self):
        res: str = ''
        if len(self.melody) != 0:
            res += 'Melody:\n'
        for event in self.melody:
            res += f'{event.event}, {event.start}\n'
        if len(self.chords) != 0:
            res += 'Chords:\n'
        for event in self.chords:
            res += f'{event.event}, {event.start}\n'
        return res.strip()

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return Timeline(self.melody.copy(), self.chords.copy())

    def get_length(self) -> Duration:
        if len(self.melody) == 0 and len(self.chords) == 0:
            return Duration(0, 1)
        if len(self.melody) == 0:
            return self.chords[-1].start + self.chords[-1].event.duration
        if len(self.chords) == 0:
            return self.melody[-1].start + self.melody[-1].event.duration
        else:
            return max(
                self.melody[-1].start + self.melody[-1].event.duration,
                self.chords[-1].start + self.chords[-1].event.duration,
            )

    def to_dict(self) -> dict[Duration, list[Note]]:
        res = {}
        for melody_event in self.melody:
            if melody_event.start not in res:
                res[melody_event.start] = []
            res[melody_event.start].append(melody_event.event)
        for chord_event in self.chords:
            if chord_event.start not in res:
                res[chord_event.start] = []
            res[chord_event.start].extend(chord_event.event.notes)
        return res


class Mode(Enum):
    """A class representing a mode"""

    MAJOR = 0
    MINOR = 1

    def get_intervals(self) -> list[int]:
        match self:
            case Mode.MAJOR:
                return [2, 2, 1, 2, 2, 2]
            case Mode.MINOR:
                return [2, 1, 2, 2, 1, 2]


class Scale(NamedTuple):
    """A class representing a scale"""

    key: Key
    mode: Mode

    def __str__(self):
        return f'{self.key} {self.mode.name}'

    def __repr__(self):
        return self.__str__()


def get_chord_function(scale: Scale, chord: Chord_in_keys) -> Chord_function:
    """Return the chord function of a chord"""
    keys_in_scale = get_keys_in_scale(scale)
    tonic_roots = [keys_in_scale[0], keys_in_scale[2], keys_in_scale[5]]
    subdominant_roots = [keys_in_scale[3], keys_in_scale[1]]
    dominant_roots = [keys_in_scale[4], keys_in_scale[2], keys_in_scale[6]]
    if chord.notes[0] in tonic_roots:
        return Chord_function.TONIC
    elif chord.notes[0] in subdominant_roots:
        return Chord_function.SUBDOMINANT
    elif chord.notes[0] in dominant_roots:
        return Chord_function.DOMINANT
    else:
        raise ValueError(f'Chord {chord} is not in scale {scale}')


def get_keys_in_scale(scale: Scale) -> list[Key]:
    """Return the keys in a scale"""
    keys = [scale.key]
    for interval in scale.mode.get_intervals():
        keys.append(Key((keys[-1].value + interval) % 12))
    return keys


class Composition(NamedTuple):
    """A class representing a composition"""

    bpm: BPM
    time_signature: Time_signature
    timeline: Timeline
    length: Duration
    scale: Scale

    def __str__(self):
        res = f'BPM: {self.bpm}\n'
        res += f'Time signature: {self.time_signature}\n'
        res += f'Scale: {self.scale}\n'
        res += f'Timeline(note, duration, start):\n{self.timeline}\n'
        return res.strip()

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return Composition(
            self.bpm,
            self.time_signature,
            self.timeline.copy(),
            self.length,
            self.scale,
        )


def read_midi_file(path: str) -> mido.MidiFile:
    """Read a midi file and return a mido.MidiFile object"""
    return mido.MidiFile(path)


def __get_clocks_per_beat(midi_file: mido.MidiFile) -> int:
    """Return the number of clocks per beat"""
    return midi_file.ticks_per_beat


def __get_bpm(midi_file: mido.MidiFile) -> BPM:
    """Return the BPM of a midi file"""
    set_tempo_message = list(
        filter(lambda m: m.type == 'set_tempo', midi_file.tracks[0])
    )
    if len(set_tempo_message) == 0:
        raise NoSetTempoMessageFound
    else:
        return mido.tempo2bpm(set_tempo_message[0].tempo)


def __get_time_signature(midi_file: mido.MidiFile) -> Time_signature:
    """Return the time signature of the midi file"""
    for msg in midi_file.tracks[0]:
        if msg.type == 'time_signature':
            return Time_signature(msg.numerator, msg.denominator)
    raise NoTimeSignatureMessageFound


def __convert_track_to_timeline(
    mono_track: mido.MidiTrack,
    clocks_per_beat: int,
    time_signature: Time_signature,
) -> Timeline:
    """Convert a mono track to a list of notes"""
    notes_in_time: list[Event_in_time[Note]] = []
    notes_start: dict[Note, Duration] = {}
    current_time = Fraction(0, 1)
    for msg in mono_track:
        current_time += Fraction(msg.time, clocks_per_beat * 4)
        match msg.type:
            case 'note_on':
                notes_start[msg.note] = duration_from_fraction(
                    current_time, time_signature
                )
            case 'note_off':
                start = notes_start.pop(msg.note)
                notes_in_time.append(
                    Event_in_time(
                        Note(
                            pitch=msg.note,
                            duration=duration_from_fraction(
                                current_time
                                - Fraction(start.numerator, start.denominator),
                                time_signature,
                            ),
                        ),
                        start,
                    )
                )
    return Timeline(notes_in_time, [])


def __get_key_from_music21_analysis(analyzed: str) -> Key:
    root_str = analyzed.split(' ')[0].upper().replace('#', '_SHARP')
    for key in Key:
        if key.name == root_str:
            return key
    raise ValueError('Could not find the key')


def __get_mode_from_music21_analysis(analyzed: str) -> Mode:
    mode_str = str(analyzed).split(' ')[1].upper()
    for mode in Mode:
        if mode.name == mode_str:
            return mode
    raise ValueError('Could not find the mode')


def __get_scale_music21(midi_file_path: str) -> Scale:
    """Return the scale of the composition"""
    song = music21.converter.parse(midi_file_path)
    analyzed = str(song.analyze('key'))
    root = __get_key_from_music21_analysis(analyzed)
    mode = __get_mode_from_music21_analysis(analyzed)
    return Scale(root, mode)


def get_composition(midi_file_path: str) -> Composition:
    midi_file = read_midi_file(midi_file_path)
    clocks = __get_clocks_per_beat(midi_file)
    time_signature = __get_time_signature(midi_file)
    timeline = __convert_track_to_timeline(
        midi_file.tracks[1], clocks, time_signature
    )
    scale = __get_scale_music21(midi_file_path)
    return Composition(
        bpm=__get_bpm(midi_file),
        time_signature=time_signature,
        scale=scale,
        timeline=timeline,
        length=timeline.get_length(),
    )


def __get_midi_from_note_list(
        notes: list[Event_in_time[Note]], clocks_per_beat: int,
) -> mido.MidiTrack:
    """Convert a notes to a midi track"""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name='Elec. Piano (Classic)',
                                  time=0))
    events: dict[int, list[Note]] = {}
    for note in notes:
        time = duration_to_clocks(note.start, clocks_per_beat)
        if time in events:
            events[time].append(note.event)
        else:
            events[time] = [note.event]
    start_times: list[int] = list(events.keys())
    note_events: dict[
        int, list[tuple[Note, Literal['note_on'] | Literal['note_off']]]] = {}
    for time in start_times:
        for event in events[time]:
            if time in note_events:
                note_events[time].append((event, 'note_on'))
            else:
                note_events[time] = [(event, 'note_on')]

            duration = duration_to_clocks(event.duration, clocks_per_beat)
            if time + duration in note_events:
                note_events[time + duration].append((event, 'note_off'))
            else:
                note_events[time + duration] = [(event, 'note_off')]

    times = list(note_events.keys())
    times.sort()
    previous_time_in_clocks = 0
    for time in times:
        on_events = list(
            filter(lambda e: e[1] == 'note_on', note_events[time])
        )
        off_events = list(
            filter(lambda e: e[1] == 'note_off', note_events[time])
        )
        for event in off_events:
            track.append(
                mido.Message(
                    'note_off',
                    note=event[0].pitch,
                    velocity=100,
                    time=time - previous_time_in_clocks,
                )
            )
            previous_time_in_clocks = time
        for event in on_events:
            track.append(
                mido.Message(
                    'note_on',
                    note=event[0].pitch,
                    velocity=50,
                    time=time - previous_time_in_clocks,
                )
            )
            previous_time_in_clocks = time
    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


def write_midi_from_track(
        track: mido.MidiTrack, clocks_per_beat: int, bpm: BPM, file_path: str
) -> None:
    """Write a midi track to a file"""
    midi_file = mido.MidiFile()
    midi_file.ticks_per_beat = clocks_per_beat
    midi_file.type = 1

    info_track = mido.MidiTrack()
    info_track.append(mido.MetaMessage('track_name', name='Info', time=0))
    info_track.append(mido.MetaMessage(
                          'time_signature', numerator=4, denominator=4))
    info_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    info_track.append(mido.MetaMessage('end_of_track', time=0))

    midi_file.tracks.append(info_track)
    midi_file.tracks.append(track)

    midi_file.save(file_path)


def get_basic_chords_in_scale(
        scale: Scale, duration: Duration) -> list[Chord_in_keys]:
    """Return the chords in a scale"""
    keys = get_keys_in_scale(scale)
    chords = []
    for i in range(0, len(keys)):
        chords.append(
            Chord_in_keys(
                notes=[
                    keys[i],
                    keys[(i + 2) % len(keys)],
                    keys[(i + 4) % len(keys)],
                ],
                duration=duration,
            )
        )
    return chords

# -------------------------- Configuration --------------------------------


POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 500
NUMBER_OF_ELITE = 30
MUTATION_PROBABILITY = 0.03
NUMBER_OF_CROSSOVERS = 20
NUMBER_OF_INDIVIDUALS_TO_CROSSOVER = 2
CHORD_DURATION = Duration(1, 2)
VERTICAL_DISSONANCE_OCTAVE_COEFFICIENT = 0.5
HORIZONTAL_DISSONANCE_OCTAVE_COEFFICIENT = 0.5
VERTICAL_DISSONANCE_COEFFICIENT = 1000
HORIZONTAL_DISSONANCE_COEFFICIENT = 1
HARMONIC_MOVEMENTS_COFFICIENT = 100
REPETITION_PENALTY = 100


# ----------------------- Genetic algorithm --------------------------------


class Individual(NamedTuple):
    composition: Composition

    def genome(self):
        return self.composition.timeline.chords

    def copy(self):
        return Individual(self.composition.copy())


class Population(NamedTuple):
    individuals: list[Individual]

    def copy(self):
        return Population(self.individuals.copy())


def pitch_difference_to_fitness(pitch_difference: int) -> float:
    """Return the fitness of a pitch difference"""
    difference_in_octave = abs(pitch_difference) % 12
    if (difference_in_octave == 0 or
            difference_in_octave == 5):
        return 1
    elif (difference_in_octave == 1 or
          difference_in_octave == 11):
        return 0.3
    elif (difference_in_octave == 2 or
          difference_in_octave == 10):
        return 0.4
    elif (difference_in_octave == 3 or
          difference_in_octave == 4 or
          difference_in_octave == 7 or
          difference_in_octave == 8 or
          difference_in_octave == 9):
        return 0.5
    elif difference_in_octave == 6:
        return 0.1
    else:
        raise ValueError('Invalid pitch difference')


def dissonance_fitness(note1: Note | Key, note2: Note | Key,
                       dissonance_octave_coefficient: float) -> float:
    """Return the fitness of an individual"""
    difference = 0
    interval = 0
    octave_difference = 0
    if isinstance(note1, Note) and isinstance(note2, Note):
        difference = abs(note1.pitch - note2.pitch)
        interval = difference % 12
        octave_difference = difference // 12
    else:
        pitch1 = 0
        pitch2 = 0
        if isinstance(note1, Key):
            pitch1 = note1.value
        elif isinstance(note1, Note):
            pitch1 = note1.pitch
        if isinstance(note2, Key):
            pitch2 = note2.value
        elif isinstance(note2, Note):
            pitch2 = note2.pitch
        difference = abs(pitch1 - pitch2)
        interval = difference % 12
        octave_difference = 0

    octave_fitness = (dissonance_octave_coefficient ** octave_difference)
    return pitch_difference_to_fitness(interval) * octave_fitness


def vertical_dissonance_fitness(individual: Individual,
                                dissonance_octave_coefficient: float) -> float:
    """Return the fitness of an individual"""
    fitness = 0
    notes_in_time = individual.composition.timeline.to_dict()
    for notes in notes_in_time.values():
        for note1 in notes:
            for note2 in notes:
                if note1 != note2:
                    fitness += dissonance_fitness(
                        note1, note2, dissonance_octave_coefficient)
    return fitness


def horizontal_dissonance_fitness(
        individual: Individual, dissonance_octave_coefficient: float) -> float:
    """Return the fitness of an individual"""
    fitness = 0
    notes_in_time = individual.composition.timeline.to_dict()
    times = list(notes_in_time.keys())
    times.sort()
    time = times[0]
    for i in range(1, len(times) - 1):
        for note1 in notes_in_time[time]:
            for note2 in notes_in_time[times[i]]:
                fitness += dissonance_fitness(
                    note1, note2, dissonance_octave_coefficient)
        time = times[i]
    return fitness


def vertical_fitness(
        individual: Individual, dissonance_octave_coefficient: float,
        dissonance_coefficient: float) -> float:
    """Return the vertical fitness of an individual"""
    fitness = 0
    fitness += dissonance_coefficient * vertical_dissonance_fitness(
        individual, dissonance_octave_coefficient)
    return fitness


def movement_prioritized(function1: Chord_function,
                         function2: Chord_function) -> bool:
    """Return if movement is prioritized(one from the list):
         dominant -> tonic
         subdominant -> dominant or tonic
         tonic -> subdominant or dominant
    """
    if function1 == Chord_function.DOMINANT:
        if function2 == Chord_function.TONIC:
            return True
    elif function1 == Chord_function.SUBDOMINANT:
        if (function2 == Chord_function.DOMINANT or
                function2 == Chord_function.TONIC):
            return True
    elif function1 == Chord_function.TONIC:
        if (function2 == Chord_function.SUBDOMINANT or
                function2 == Chord_function.DOMINANT):
            return True
    return False


def harmonic_movements_fitness(individual: Individual) -> float:
    """Return the fitness of an individual based on the harmonic movements"""
    fitness = 0
    notes_in_time = individual.composition.timeline.chords
    chord_dict: dict[Duration, Chord_in_keys] = {}
    for chord in notes_in_time:
        chord_dict[chord.start] = chord.event

    times = list(chord_dict.keys())
    times.sort()
    if len(times) == 0:
        return 0
    previous_chord_function = get_chord_function(individual.composition.scale,
                                                 chord_dict[times[0]])
    for i in range(1, len(times) - 1):
        next_chord_function = get_chord_function(individual.composition.scale,
                                                 chord_dict[times[i]])
        if movement_prioritized(previous_chord_function, next_chord_function):
            fitness += 1

    return fitness


def get_repetition_penalty(individual: Individual) -> float:
    """Return the fitness of an individual based on the repetition penalty"""
    fitness = 0
    notes_in_time = individual.composition.timeline.chords
    chord_dict: dict[Duration, Chord_in_keys] = {}
    for chord in notes_in_time:
        chord_dict[chord.start] = chord.event

    times = list(chord_dict.keys())
    times.sort()
    if len(times) == 0:
        return 0

    # calculate sum of distances between the same chords and sum up for all of
    # the chords
    for i in range(0, len(times) - 1):
        for j in range(i + 1, len(times)):
            if chord_dict[times[i]] == chord_dict[times[j]]:
                fraction1 = times[i].numerator / times[i].denominator
                fraction2 = times[j].numerator / times[j].denominator
                fitness += 1/abs(fraction1 - fraction2)

    return -fitness


def horizontal_fitness(
        individual: Individual,
        dissonance_octave_coefficient: float,
        dissonance_coefficient: float,
        harmonic_movements_cofficient: float,
        repetition_penalty: float) -> float:
    """Return the horizontal fitness of an individual"""
    fitness = 0
    fitness += dissonance_coefficient * horizontal_dissonance_fitness(
        individual, dissonance_octave_coefficient)
    fitness += harmonic_movements_cofficient * harmonic_movements_fitness(
            individual)
    fitness += repetition_penalty * get_repetition_penalty(individual)
    # print(repetition_penalty * get_repetition_penalty(individual))
    return fitness


def fitness(individual: Individual) -> float:
    """Return the fitness of an individual
        The more, the better"""
    fitness = 0
    fitness += vertical_fitness(
        individual,
        VERTICAL_DISSONANCE_OCTAVE_COEFFICIENT,
        VERTICAL_DISSONANCE_COEFFICIENT
    )
    fitness += horizontal_fitness(
        individual,
        HORIZONTAL_DISSONANCE_OCTAVE_COEFFICIENT,
        HORIZONTAL_DISSONANCE_COEFFICIENT,
        HARMONIC_MOVEMENTS_COFFICIENT,
        REPETITION_PENALTY
    )
    return fitness


# TODO: maybe chop some chords into 2-4 ones
# https://youtu.be/Kv2lIrr0VWA
def mutate(individual: Individual, probability: float) -> Individual:
    """Mutate an individual"""
    genome = individual.genome().copy()
    for i in range(0, len(genome)):
        if random() < probability:
            chords = get_basic_chords_in_scale(individual.composition.scale,
                                               genome[i].event.duration)
            chord_number = randint(0, len(chords) - 1)
            genome[i] = Event_in_time(chords[chord_number], genome[i].start)

    new_individual = Individual(Composition(individual.composition.bpm,
                                individual.composition.time_signature,
                                Timeline([], genome),
                                individual.composition.length,
                                individual.composition.scale))
    # print("Mutated")
    # print(individual)
    # print(new_individual)
    return new_individual


def crossover(individuals: list[Individual],
              number_of_crossovers: int) -> list[Individual]:
    """Crossover list of individuals"""
    if len(individuals) == 0:
        raise ValueError("Individuals list is empty, not able to crossover")
    new_individuals = individuals.copy()
    for _ in range(0, number_of_crossovers):
        individual1 = choice(new_individuals)
        individual2 = choice(new_individuals)
        genome1 = individual1.genome().copy()
        genome2 = individual2.genome().copy()
        crossover_point = randint(0, len(genome1) - 1)
        genome1[crossover_point:], genome2[crossover_point:] = (
            genome2[crossover_point:],
            genome1[crossover_point:],
        )
    return new_individuals


def get_best(
    number_of_elite: int, population: Population
) -> list[Individual]:
    """Return the elite of a population"""
    return sorted(population.individuals, key=lambda i: fitness(i))[
        -number_of_elite:
    ]


def get_worst(
    number_of_elite: int, population: Population
) -> list[Individual]:
    """Return the not elite of a population"""
    return sorted(population.individuals, key=lambda i: fitness(i))[
        :number_of_elite
    ]


def get_random_individual(composition: Composition) -> Individual:
    """Return a random individual"""
    chord_duration = CHORD_DURATION
    composition_duration = composition.length
    composition_with_random_chords = composition.copy()
    chords = get_basic_chords_in_scale(composition.scale, chord_duration)
    for i in range(0, int(composition_duration / chord_duration)):
        chord_number = randint(0, len(chords) - 1)
        composition_with_random_chords.timeline.chords.append(
            Event_in_time[Chord_in_keys](
                chords[chord_number], chord_duration * i)
        )
    return Individual(composition_with_random_chords).copy()


def average_population_fitness(population: Population) -> float:
    """Return the average fitness of a population"""
    fitness_sum = 0
    for individual in population.individuals:
        fitness_sum += fitness(individual)
    return fitness_sum / len(population.individuals)


def best_population_fitness(population: Population) -> float:
    """Return the best fitness of a population"""
    return fitness(max(population.individuals, key=lambda i: fitness(i)))


def get_initial_population(
    number_of_individuals: int, composition: Composition
) -> Population:
    """Return the initial population"""
    individuals = [
        get_random_individual(composition)
        for _ in range(number_of_individuals)
    ]
    return Population(individuals)


def get_best_individual(population: Population) -> Individual:
    """Return the best individual"""
    return sorted(population.individuals, key=lambda i: fitness(i))[-1]


def get_next_generation(population: Population,
                        elite_number: int,
                        mutation_probability: float,
                        crossover_number: int,
                        number_of_individuals_to_crossover: int) -> Population:
    """Return the next generation"""
    elite = get_best(elite_number, population)
    crossover_individuals = get_best(number_of_individuals_to_crossover,
                                     population)
    new_individuals = crossover(crossover_individuals, crossover_number).copy()
    new_individuals.extend(elite.copy())
    new_individuals = [
        mutate(i, mutation_probability) for i in new_individuals]

    # print("Size: ", len(new_individuals))
    return Population(new_individuals).copy()


def genetic_algorithm(
    initial_population: Population,
    number_of_generations: int,
    elite_number: int,
    mutation_probability: float,
    crossover_number: int,
    number_of_individuals_to_crossover: int
) -> Population:
    """Return the best individual"""
    generation = initial_population
    for i in range(number_of_generations):
        generation = get_next_generation(generation, elite_number,
                                         mutation_probability,
                                         crossover_number,
                                         number_of_individuals_to_crossover)
        print(f"{i} best fitness: {best_population_fitness(generation)}")
    return generation


def chords_to_notes(chords: list[
                    Event_in_time[Chord_in_keys]],
                    octave: int) -> list[Event_in_time[Note]]:
    """Return a list of notes from a list of chords"""
    notes: list[Event_in_time[Note]] = []
    for chord in chords:
        for note in chord.event.notes:
            notes.append(Event_in_time[Note](
                Note(
                    note.value + octave * 12,
                    chord.event.duration,
                ),
                chord.start,
            ))
    return notes


parser = argparse.ArgumentParser(description="Genetic algorithm")
parser.add_argument('path', type=str, help="Path to the midi file")
# parser.add_argument(
# 'number_of_generations', type=int,  help="Number of generations")


def main():
    args = parser.parse_args()
    input_path = ''
    if os.path.isabs(args.path):
        input_path = args.path
    else:
        input_path = os.getcwd() + '/' + args.path

    output_path = input_path.replace('.mid', '_output.mid')
    composition = get_composition(input_path)
    last_population = genetic_algorithm(
        get_initial_population(POPULATION_SIZE, composition),
        NUMBER_OF_GENERATIONS,
        NUMBER_OF_ELITE,
        MUTATION_PROBABILITY,
        NUMBER_OF_CROSSOVERS,
        NUMBER_OF_INDIVIDUALS_TO_CROSSOVER
    )
    best_individual = get_best_individual(last_population)
    midi = mido.MidiFile(input_path)
    clocks_per_beat = midi.ticks_per_beat
    chords_in_keys = best_individual.genome()
    chords = chords_to_notes(chords_in_keys, 4)
    notes = composition.timeline.melody + chords
    track = __get_midi_from_note_list(notes, clocks_per_beat)
    bpm = composition.bpm
    write_midi_from_track(track, clocks_per_beat, bpm, output_path)


if __name__ == '__main__':
    main()
