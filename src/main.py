import os
from dataclasses import astuple, dataclass
from enum import Enum
from fractions import Fraction
from math import gcd
from random import randint, random, choice
from typing import Generic, NamedTuple, TypeVar

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


class Chord(NamedTuple):
    """A class representing a chord"""

    notes: list[Pitch] | list[Key]
    duration: Duration

    def __str__(self):
        return f'{self.notes}'

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return Chord(self.notes.copy(), self.duration)


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
    chords: list[Event_in_time[Chord]]

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


def get_composition(midi_file_path: str):
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


def get_basic_chords_in_scale(scale: Scale, duration: Duration) -> list[Chord]:
    """Return the chords in a scale"""
    keys = get_keys_in_scale(scale)
    chords = []
    for i in range(0, len(keys)):
        chords.append(
            Chord(
                notes=[
                    keys[i],
                    keys[(i + 2) % len(keys)],
                    keys[(i + 4) % len(keys)],
                ],
                duration=duration,
            )
        )
    return chords


class Individual(NamedTuple):
    composition: Composition

    def genome(self):
        return self.composition.timeline.chords


class Population(NamedTuple):
    individuals: list[Individual]

    def copy(self):
        return Population(self.individuals.copy())


def fitness(individual: Individual) -> float:
    """Return the fitness of an individual"""
    # TODO: todo
    # - Add checking for simultaneous and consecutive dissonance notes
    # - Check for number of similar notes in chord and melody
    #   - Hovewer, this coefficient shouldn't be big, otherwise it will sound
    #   bad https://youtu.be/TqJ8M2GfenI
    #   - When checking it also chords on сильных долях should have more
    #   priority https://youtu.be/TqJ8M2GfenI, but 2 notes in the same chord
    #   на слабых долях should have more impact then one note on сильной доле
    # - If there wasn't tonic chord in more than 8/4 there should be penalty
    # Everything can be separated into 2 parameters how well it sounds
    # vertically and how well it sounds horizontally

    # This movemements should have really high priority:
    # dominant -> tonic
    # subdominant -> dominant or tonic
    # tonic -> subdominant or dominant
    fitness = random()
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
    return Individual(Composition(individual.composition.bpm,
                                  individual.composition.time_signature,
                                  Timeline([], genome),
                                  individual.composition.length,
                                  individual.composition.scale))


def crossover(individuals: list[Individual],
              number_of_crossovers: int) -> list[Individual]:
    """Crossover list of individuals"""
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
    chord_duration = duration_from_fraction(Fraction(1, 4),
                                            composition.time_signature)
    composition_duration = composition.length
    composition_with_random_chords = composition.copy()
    chords = get_basic_chords_in_scale(composition.scale, chord_duration)
    for i in range(0, int(composition_duration / chord_duration)):
        chord_number = randint(0, len(chords) - 1)
        composition_with_random_chords.timeline.chords.append(
            Event_in_time[Chord](chords[chord_number], chord_duration * i)
        )
    return Individual(composition_with_random_chords)


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
    not_elite = Population(get_worst(elite_number, population))
    crossover_individuals = get_best(number_of_individuals_to_crossover,
                                     not_elite)
    not_crossed_over_individuals = get_best(number_of_individuals_to_crossover,
                                            not_elite)

    new_individuals = crossover(crossover_individuals, crossover_number)
    new_individuals.extend(not_crossed_over_individuals)

    for i in range(0, len(new_individuals)):
        new_individuals[i] = mutate(new_individuals[i], mutation_probability)

    new_individuals.extend(elite)
    return Population(new_individuals)


def genetic_algorithm(
    initial_population: Population,
    number_of_generations: int,
    elite_number: int,
    mutation_probability: float,
    crossover_number: int,
    number_of_individuals_to_crossover: int
) -> Population:
    """Return the best individual"""
    generations = [initial_population]
    for _ in range(number_of_generations):
        next = get_next_generation(generations[-1], elite_number,
                                   mutation_probability,
                                   crossover_number,
                                   number_of_individuals_to_crossover)
        generations.append(next)
    return generations[-1]


def main():
    MIDI_FILE_PATH = os.getcwd() + '/resources/barbiegirl_mono.mid'
    composition = get_composition(MIDI_FILE_PATH)
    last_population = genetic_algorithm(
        get_initial_population(100, composition),
        1000, 2, 0.1, 2, 2,)
    best_individual = get_best_individual(last_population)
    print(best_individual.composition.timeline.chords)


if __name__ == '__main__':
    main()
