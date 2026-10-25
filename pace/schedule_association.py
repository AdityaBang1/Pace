"""Clause and token association helpers for plain-text schedule extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


def _clause_boundaries(text: str) -> list[int]:
    """Find punctuation boundaries, excluding periods used inside supported abbreviations."""
    weekday_abbreviations = {"mon", "tue", "tues", "wed", "weds", "thu", "thur", "thurs", "fri", "sat", "sun"}
    period_boundaries = []
    for period in re.finditer(r"\.", text):
        before = text[:period.start()].rstrip()
        after = text[period.end():]
        preceding_word = re.search(r"([A-Za-z]+)$", before)
        if preceding_word and preceding_word.group(1).casefold() in weekday_abbreviations and re.match(
            r"\s*,?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4}-|\d{1,2}/)",
            after,
            re.I,
        ):
            continue
        if re.search(r"\b[ap]$", before, re.I) and re.match(r"m\.", after, re.I):
            continue
        if re.search(r"\b[ap]\.m$", before, re.I):
            next_word = re.match(r"\s+([A-Za-z]+)", after)
            if not next_word or not next_word.group(1)[0].isupper():
                continue
        period_boundaries.append(period.start())
    return sorted([
        *(match.start() for match in re.finditer(r"[;!?]", text)),
        *period_boundaries,
    ])


def clause_ranges(text: str) -> list[tuple[int, int]]:
    """Return trimmed source spans for clauses, retaining empty clauses to align IDs."""
    ranges = []
    start = 0
    for end in _clause_boundaries(text):
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        ranges.append((left, right))
        start = end + 1
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    ranges.append((left, right))
    return ranges


def clause_map(text: str, tokens: list[re.Match[str]]) -> dict[int, int]:
    """Map each source token's start offset to its clause ID in one ordered pass."""
    boundaries = _clause_boundaries(text)
    ordered_tokens = sorted(enumerate(tokens), key=lambda item: item[1].start())
    clauses = [0] * len(tokens)
    clause_id = boundary_index = 0
    for token_index, token in ordered_tokens:
        while boundary_index < len(boundaries) and boundaries[boundary_index] < token.start():
            clause_id += 1
            boundary_index += 1
        clauses[token_index] = clause_id
    return {token.start(): clauses[index] for index, token in enumerate(tokens)}


def nearest_match(match: re.Match[str], candidates: list[re.Match[str]]) -> re.Match[str]:
    """Return the candidate whose center is closest to the given match."""
    center = (match.start() + match.end()) / 2
    return min(candidates, key=lambda item: abs((item.start() + item.end()) / 2 - center))


def associated_nearest(
    token: re.Match[str], candidates: list[re.Match[str]], clause_by_start: dict[int, int],
) -> re.Match[str] | None:
    """Associate to the nearest candidate in the token's clause."""
    clause = clause_by_start[token.start()]
    local = [item for item in candidates if clause_by_start[item.start()] == clause]
    return nearest_match(token, local) if local else None


def date_time_match(
    date_match: re.Match[str], dates: list[re.Match[str]], times: list[re.Match[str]],
    clause_by_start: dict[int, int],
) -> tuple[re.Match[str] | None, str]:
    """Choose a local time candidate only if no other date claims the same slot."""
    date_clause = clause_by_start[date_match.start()]
    local_dates = [item for item in dates if clause_by_start[item.start()] == date_clause]
    local_times = [item for item in times if clause_by_start[item.start()] == date_clause]
    if not local_times:
        return None, "association_unconfirmed"
    if len(local_dates) != len(local_times):
        return associated_nearest(date_match, local_times, clause_by_start), "association_ambiguous"
    time_match = associated_nearest(date_match, local_times, clause_by_start)
    if time_match is None:
        return None, "association_unconfirmed"
    competing_dates = [
        other for other in local_dates
        if other is not date_match and associated_nearest(other, local_times, clause_by_start) is time_match
    ]
    if competing_dates:
        return time_match, "association_ambiguous"
    return time_match, "association_inferred"


def date_weekday_matches(
    date_match: re.Match[str], dates: list[re.Match[str]], weekdays: list[re.Match[str]],
    clause_by_start: dict[int, int],
) -> tuple[list[re.Match[str]], list[re.Match[str]], str | None]:
    """Associate weekdays only within a clause and when their ownership is unambiguous."""
    clause = clause_by_start[date_match.start()]
    clause_dates = [item for item in dates if clause_by_start[item.start()] == clause]
    clause_weekdays = [item for item in weekdays if clause_by_start[item.start()] == clause]
    if not clause_weekdays:
        return [], [], None
    if len(clause_dates) == 1:
        return clause_weekdays, clause_weekdays, None
    if len(clause_weekdays) != len(clause_dates):
        return [], clause_weekdays, "weekday_date_association_ambiguous"

    assigned = nearest_match(date_match, clause_weekdays)
    if any(
        other is not date_match and nearest_match(other, clause_weekdays) is assigned
        for other in clause_dates
    ):
        return [], clause_weekdays, "weekday_date_association_ambiguous"
    return [assigned], [assigned], "weekday_date_association_inferred"


@dataclass(frozen=True)
class DatedEventGroup:
    """A clause-local date candidate group with an explicit association state."""

    date_candidates: tuple[re.Match[str], ...]
    start: int
    end: int
    clause_id: int
    cue: re.Match[str] | None = None


def dated_event_groups(
    line: str,
    dates: list[re.Match[str]],
    clauses: list[tuple[int, int]],
    clause_by_start: dict[int, int],
    event_cue_pattern: Pattern[str],
    anchor_pattern: Pattern[str],
) -> list[DatedEventGroup]:
    """Own cue/date grouping and preserve uncertainty when pre-date cues compete."""
    dates_by_clause: dict[int, list[re.Match[str]]] = {}
    for date in dates:
        dates_by_clause.setdefault(clause_by_start[date.start()], []).append(date)

    groups = []
    for clause_id, clause_dates in dates_by_clause.items():
        clause_start, clause_end = clauses[clause_id]
        cues_before_first_date = list(
            anchor_pattern.finditer(line, clause_start, clause_dates[0].start())
        )
        all_clause_cues = list(anchor_pattern.finditer(line, clause_start, clause_end))
        if cues_before_first_date and len(all_clause_cues) == len(cues_before_first_date):
            if len(cues_before_first_date) > 1:
                for index, cue in enumerate(cues_before_first_date):
                    start = clause_start if index == 0 else cue.start()
                    groups.append(DatedEventGroup(
                        date_candidates=tuple(clause_dates),
                        start=start,
                        end=cue.end(),
                        clause_id=clause_id,
                        cue=cue,
                    ))
                continue
            if len(clause_dates) > 1:
                cue = cues_before_first_date[0]
                start, end = clause_start, clause_dates[0].start()
                connector = re.search(
                    r"\s+(?:on|are|is|was|were|will\s+be|due|scheduled\s+for)\s*$",
                    line[start:end],
                    re.I,
                )
                if connector:
                    end = start + connector.start()
                while end > start and line[end - 1] in " \t:–—-":
                    end -= 1
                groups.append(DatedEventGroup(
                    date_candidates=tuple(clause_dates),
                    start=start,
                    end=end,
                    clause_id=clause_id,
                    cue=cue,
                ))
                continue

        splits = []
        for previous, current in zip(clause_dates, clause_dates[1:]):
            cue = next(event_cue_pattern.finditer(line, previous.end(), current.start()), None)
            if cue:
                splits.append((current, cue.start()))

        group_start = clause_start
        group_dates = []
        split_index = 0
        for date in clause_dates:
            if split_index < len(splits) and splits[split_index][0] is date:
                cue_start = splits[split_index][1]
                previous_end = group_dates[-1].end()
                separator = re.search(r"(?:,?\s+(?:and|or)\s+|,\s*)$", line[previous_end:cue_start], re.I)
                group_end = previous_end + separator.start() if separator else cue_start
                while group_end > group_start and line[group_end - 1].isspace():
                    group_end -= 1
                groups.append(DatedEventGroup(
                    date_candidates=tuple(group_dates),
                    start=group_start,
                    end=group_end,
                    clause_id=clause_id,
                ))
                group_start = cue_start
                group_dates = []
                split_index += 1
            group_dates.append(date)
        groups.append(DatedEventGroup(
            date_candidates=tuple(group_dates),
            start=group_start,
            end=clause_end,
            clause_id=clause_id,
        ))
    return groups


def assigned_groups(
    anchors: list[re.Match[str]], slots: list[re.Match[str]], text: str,
    schedule_cue_pattern: Pattern[str], clause_by_start: dict[int, int],
) -> tuple[dict[int, list[re.Match[str]]], str]:
    """Associate recurring weekday anchors with time slots under conservative count rules."""
    assignments = {index: [] for index in range(len(slots))}
    anchor_clauses = {clause_by_start[anchor.start()] for anchor in anchors}
    slot_clauses = {clause_by_start[slot.start()] for slot in slots}
    if len(anchors) == 1 and len(slots) > 1:
        anchor = anchors[0]
        if slot_clauses != {clause_by_start[anchor.start()]}:
            return assignments, "association_unconfirmed"
        cue_matches = list(schedule_cue_pattern.finditer(text))
        prior_cue = next((cue for cue in reversed(cue_matches) if cue.start() < anchor.start()), None)
        if prior_cue:
            between_cue_and_anchor = text[prior_cue.end():anchor.start()]
            if not re.search(r"\b(?:and|or)\b|,|/", between_cue_and_anchor):
                return {index: list(anchors) for index in assignments}, "multiple_slots_for_single_group"
        return assignments, "association_ambiguous"
    if len(anchors) == len(slots):
        if anchor_clauses == slot_clauses and len(anchor_clauses) == 1:
            used_slots = set()
            for anchor in anchors:
                slot = nearest_match(anchor, slots)
                slot_index = slots.index(slot)
                if slot_index in used_slots:
                    return {index: [] for index in assignments}, "association_ambiguous"
                used_slots.add(slot_index)
                assignments[slot_index].append(anchor)
            return assignments, "association_inferred"
        if len(anchor_clauses) != len(anchors) or len(slot_clauses) != len(slots):
            return assignments, "association_ambiguous"
        used_slots = set()
        for anchor in anchors:
            slot = associated_nearest(anchor, slots, clause_by_start)
            if slot is None:
                return assignments, "association_unconfirmed"
            slot_index = slots.index(slot)
            if slot_index in used_slots:
                return {index: [] for index in assignments}, "association_ambiguous"
            used_slots.add(slot_index)
            assignments[slot_index].append(anchor)
        if len(used_slots) != len(slots):
            return {index: [] for index in assignments}, "association_ambiguous"
        return assignments, "association_inferred"
    if len(anchors) > len(slots) and anchor_clauses == slot_clauses:
        for slot_index, slot in enumerate(slots):
            local_anchors = [anchor for anchor in anchors if clause_by_start[anchor.start()] == clause_by_start[slot.start()]]
            if len(local_anchors) == 1:
                assignments[slot_index].extend(local_anchors)
        if all(assignments.values()):
            return assignments, "multiple_days_for_time_slots"
    return assignments, "association_ambiguous"
