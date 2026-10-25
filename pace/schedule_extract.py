"""Conservative plain-text schedule candidate extraction with source offsets."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from .schedule_association import (
    assigned_groups as _assigned_groups,
    clause_map,
    clause_ranges,
    date_time_match,
    dated_event_groups,
    date_weekday_matches as _date_weekday_matches,
)

DATE_RE = re.compile(
    r"(?<!\d)(?P<iso_year>\d{4})-(?P<iso_month>\d{1,2})-(?P<iso_day>\d{1,2})(?!\d)"
    r"|(?<!\d)(?P<num_month>\d{1,2})/(?P<num_day>\d{1,2})/(?P<num_year>\d{4})(?!/\d)"
    r"|(?<!\d)(?P<short_month>\d{1,2})/(?P<short_day>\d{1,2})(?!/\d)"
    r"|\b(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<name_day>\d{1,2})"
    r"(?:\s*[-–—]\s*(?P<name_range_end>\d{1,2}))?"
    r"(?:,?\s+(?P<name_year>\d{4}))?\b",
    re.I,
)


def _time_token(prefix: str) -> str:
    return (
        rf"(?P<{prefix}_hour>\d{{1,2}})"
        rf"(?::(?P<{prefix}_minute>\d{{2}}))?\s*"
        rf"(?P<{prefix}_meridiem>[ap]\.?(?:m)\.?)?"
    )


TIME_RANGE_RE = re.compile(
    r"(?<!\d)" + _time_token("start") + r"\s*(?:-|–|—|\bto\b)\s*"
    + _time_token("end") + r"(?!\d)",
    re.I,
)
TIME_POINT_RE = re.compile(
    r"(?<!\d)(?P<hour>\d{1,2})"
    r"(?=(?::\d{2}|\s*[ap]\.?(?:m)\.?(?![A-Za-z])))"
    r"(?::(?P<minute>\d{2}))?\s*(?P<meridiem>[ap]\.?(?:m)\.?)?(?![A-Za-z])",
    re.I,
)
DAY_RE = re.compile(
    r"(?<![A-Za-z])(?P<compact>TuTh|TTh|MWF|MW|WF)(?![A-Za-z])"
    r"|(?<![A-Za-z])(?P<slash>[MTWRF](?:\s*/\s*[MTWRF])+)(?![A-Za-z])"
    r"|(?<![A-Za-z])(?P<weekday>Wednesdays?|Thursdays?|Tuesdays?|Saturdays?|Sundays?|Mondays?|Fridays?"
    r"|Weds?|Thurs?|Tues?|Mon|Fri|Sat|Sun)(?![A-Za-z])",
    re.I,
)
SCHEDULE_CUE_RE = re.compile(
    r"\b(?:class(?:es)?|lectures?|seminars?|meetings?|meets|sessions?|office\s+hours|"
    r"schedule|timetable|weekly|recurring|labs?|tutorials?)\b",
    re.I,
)
# Compound cues must match as a whole (e.g. "final exam"), not as "final" + "exam".
DATED_EVENT_CUE_RE = re.compile(
    r"\b(?:final\s+exams?|final\s+examinations?|midterm\s+exams?|midterm\s+examinations?|"
    r"exam(?:ination)?s?|quiz(?:zes)?|tests?|midterms?|finals?|due|deadline|assignments?|projects?|"
    r"papers?|presentations?|assessments?|events?|classes|lectures?|meetings?|sessions?|labs?|"
    r"week|schedule|office\s+hours)\b",
    re.I,
)
DATED_EVENT_ANCHOR_RE = re.compile(
    r"\b(?:final\s+exams?|final\s+examinations?|midterm\s+exams?|midterm\s+examinations?|"
    r"exam(?:ination)?s?|quiz(?:zes)?|tests?|midterms?|finals?|assignments?|projects?|papers?|"
    r"presentations?|assessments?|events?|classes|lectures?|meetings?|sessions?|labs?|office\s+hours)\b",
    re.I,
)

MONTHS = {name.casefold(): month for month, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
    start=1,
)}
DAY_NAMES = {
    "monday": "MO", "mon": "MO", "tuesday": "TU", "tue": "TU", "tues": "TU",
    "wednesday": "WE", "wed": "WE", "weds": "WE", "thursday": "TH", "thu": "TH",
    "thur": "TH", "thurs": "TH", "friday": "FR", "fri": "FR",
    "saturday": "SA", "sat": "SA", "sunday": "SU", "sun": "SU",
}
COMPACT_DAYS = {"m": "MO", "t": "TU", "w": "WE", "r": "TH", "f": "FR"}
COMPACT_GROUPS = {
    "mwf": ["MO", "WE", "FR"], "mw": ["MO", "WE"], "wf": ["WE", "FR"],
    "tth": ["TU", "TH"], "tuth": ["TU", "TH"],
}


def _date_value(match: re.Match[str]) -> tuple[dict[str, Any], list[str]]:
    groups = match.groupdict()
    if groups["iso_year"]:
        year, month, day = int(groups["iso_year"]), int(groups["iso_month"]), int(groups["iso_day"])
    elif groups["num_month"]:
        year, month, day = int(groups["num_year"]), int(groups["num_month"]), int(groups["num_day"])
    elif groups["short_month"]:
        year, month, day = None, int(groups["short_month"]), int(groups["short_day"])
    else:
        year = int(groups["name_year"]) if groups["name_year"] else None
        month = MONTHS[groups["month_name"].casefold()]
        day = int(groups["name_day"])

    ambiguities = []
    validation_year = year if year is not None else 2000
    try:
        dt.date(validation_year, month, day)
        valid = True
    except ValueError:
        valid = False
    if year is None:
        ambiguities.append("year_missing")
    if groups["num_month"] or groups["short_month"]:
        ambiguities.append("numeric_date_order_locale_assumed_month_first")
    if not valid:
        ambiguities.append("invalid_calendar_date")
    value = {
        "year": year,
        "month": month,
        "day": day,
        "iso_date": dt.date(year, month, day).isoformat() if valid and year is not None else None,
    }
    range_end = groups.get("name_range_end")
    if range_end:
        end_day = int(range_end)
        try:
            dt.date(validation_year, month, end_day)
            end_valid = end_day >= day
        except ValueError:
            end_valid = False
        value["range_end"] = {
            "year": year,
            "month": month,
            "day": end_day,
            "iso_date": dt.date(year, month, end_day).isoformat() if end_valid and year is not None else None,
        }
        ambiguities.append("date_range_candidate")
        if not end_valid:
            ambiguities.append("invalid_date_range_end")
    return value, ambiguities


def _time_matches(pattern: re.Pattern[str], line: str, dates: list[re.Match[str]]) -> list[re.Match[str]]:
    """Ignore apparent time tokens that overlap literal numeric date components."""
    return [
        match for match in pattern.finditer(line)
        if not any(match.start() < date.end() and date.start() < match.end() for date in dates)
    ]


def _time_slots(line: str, dates: list[re.Match[str]]) -> list[re.Match[str]]:
    """Collect time ranges and non-overlapping points across every clause."""
    ranges = _time_matches(TIME_RANGE_RE, line, dates)
    points = _time_matches(TIME_POINT_RE, line, dates)
    points = [
        point for point in points
        if not any(point.start() < item.end() and item.start() < point.end() for item in ranges)
    ]
    return sorted([*ranges, *points], key=lambda match: match.start())


def _time_value(match: re.Match[str]) -> tuple[dict[str, str | None], list[str], str]:
    if "start_hour" in match.groupdict():
        value, issues = _parse_time_range(match)
        return value, issues, "time_range"
    start, issue = _parse_clock(match.group("hour"), match.group("minute"), match.group("meridiem"))
    return {"start": start, "end": None}, [issue] if issue else ["end_time_missing"], "time"


def _trim_terminal_punctuation(match: re.Match[str]) -> tuple[int, int]:
    start, end = match.start(), match.end()
    while end > start and match.string[end - 1] in ".,;!?":
        end -= 1
    return start, end


def _time_evidence(text: str, line_start: int, match: re.Match[str], key: str) -> dict[str, Any]:
    start, end = _trim_terminal_punctuation(match)
    return {key: _evidence(text, line_start + start, line_start + end)}


def _parse_clock(hour_text: str, minute_text: str | None, meridiem_text: str | None) -> tuple[str | None, str | None]:
    hour = int(hour_text)
    minute = int(minute_text or "0")
    if minute > 59:
        return None, "invalid_time"
    if meridiem_text:
        meridiem = meridiem_text.casefold().replace(".", "")
        if hour < 1 or hour > 12:
            return None, "invalid_time"
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    elif hour <= 12:
        return None, "meridiem_missing"
    elif hour > 23:
        return None, "invalid_time"
    return f"{hour:02d}:{minute:02d}", None


def _parse_time_range(match: re.Match[str]) -> tuple[dict[str, str | None], list[str]]:
    groups = match.groupdict()
    start_meridiem, end_meridiem = groups["start_meridiem"], groups["end_meridiem"]
    inferred_cross_noon = False
    if start_meridiem and not end_meridiem:
        end_meridiem = start_meridiem
    elif end_meridiem and not start_meridiem:
        start_hour, end_hour = int(groups["start_hour"]), int(groups["end_hour"])
        if start_hour == 12:
            start_meridiem = end_meridiem
        elif end_hour < start_hour or (end_hour == 12 and start_hour < 12):
            start_meridiem = "am" if end_meridiem.casefold().startswith("p") else "pm"
            inferred_cross_noon = True
        else:
            start_meridiem = end_meridiem
    elif not start_meridiem and not end_meridiem and int(groups["start_hour"]) <= 12 < int(groups["end_hour"]):
        # Infer the common 9:00-13:00 convention, but retain an ambiguity for review.
        start_meridiem = "am"
        end_meridiem = "pm"
        inferred_cross_noon = True
    start, start_error = _parse_clock(groups["start_hour"], groups["start_minute"], start_meridiem)
    end, end_error = _parse_clock(groups["end_hour"], groups["end_minute"], end_meridiem)
    if start_error == "meridiem_missing" or end_error == "meridiem_missing":
        if not groups["start_meridiem"] and not groups["end_meridiem"]:
            issues = {"meridiem_missing"}
            return {"start": None, "end": None}, sorted(issues)
    issues = {issue for issue in (start_error, end_error) if issue}
    if inferred_cross_noon:
        issues.add("meridiem_inferred_for_crossing_noon_range")
    return {"start": start, "end": end}, sorted(issues)


def _days_from_match(match: re.Match[str]) -> list[str]:
    groups = match.groupdict()
    if groups["compact"]:
        return COMPACT_GROUPS[groups["compact"].casefold()]
    if groups["slash"]:
        return [COMPACT_DAYS[token.casefold()] for token in re.findall(r"[MTWRF]", groups["slash"], re.I)]
    weekday = groups["weekday"].casefold().removesuffix("s")
    return [DAY_NAMES[weekday]]


def _evidence(text: str, start: int, end: int) -> dict[str, Any]:
    return {"text": text[start:end], "start": start, "end": end}


def _lines(text: str):
    offset = 0
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        leading = len(body) - len(body.lstrip())
        end = len(body.rstrip())
        if leading < end:
            yield offset + leading, offset + end
        offset += len(raw)


def _event(event_type: str, text: str, start: int, end: int, normalized: dict[str, Any],
           field_evidence: dict[str, Any], ambiguities: list[str]) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "summary": text[start:end],
        "normalized": normalized,
        "evidence": _evidence(text, start, end),
        "field_evidence": field_evidence,
        "ambiguities": sorted(set(ambiguities)),
        "requires_review": True,
        "confidence": None,
        "confidence_calibrated": False,
        "confidence_semantics": "uncalibrated_literal_schedule_candidate",
    }


def _serialize_dated_event_candidate(
    *,
    start: int,
    end: int,
    date_candidates: tuple[re.Match[str], ...],
    date_association_status: str,
    normalized: dict[str, Any],
    field_evidence: dict[str, Any],
    ambiguities: list[str],
    text: str,
    line_start: int,
) -> dict[str, Any]:
    """Serialize resolved or ambiguous dates with one predictable evidence shape."""
    date_evidence = [
        {
            **_evidence(text, line_start + date.start(), line_start + date.end()),
            "normalized": _date_value(date)[0],
        }
        for date in date_candidates
    ]
    field_evidence = dict(field_evidence)
    field_evidence["dates"] = date_evidence
    if date_association_status == "resolved":
        date = date_evidence[0]
        # Retain the original singular field for existing consumers.
        field_evidence.setdefault(
            "date", {key: date[key] for key in ("text", "start", "end")}
        )

    result = _event(
        "dated_event_candidate",
        text,
        line_start + start,
        line_start + end,
        normalized,
        field_evidence,
        ambiguities,
    )
    result["date_association_status"] = date_association_status
    return result


def _recurring_events(text: str, line_start: int, line_end: int) -> list[dict[str, Any]]:
    """Extract recurring weekday/time candidates from one date-free source clause."""
    events = []
    line = text[line_start:line_end]
    days = list(DAY_RE.finditer(line))
    time_slots = _time_slots(line, [])
    if not (days and time_slots):
        return events

    normalized = {
        "date": None,
        "weekdays": list(dict.fromkeys(day for match in days for day in _days_from_match(match))),
        "start_time": None,
        "end_time": None,
    }
    ambiguities = []
    if not SCHEDULE_CUE_RE.search(line):
        ambiguities.append("schedule_context_unconfirmed")
    evidence = {"weekdays": [
        {**_evidence(text, line_start + match.start(), line_start + match.end()), "days": _days_from_match(match)}
        for match in days
    ]}
    if len(time_slots) > 1:
        clause_tokens = [*days, *time_slots]
        clause_by_start = clause_map(line, clause_tokens)
        assigned_days, association_status = _assigned_groups(
            days, time_slots, line, SCHEDULE_CUE_RE, clause_by_start,
        )
        association_ambiguity = {
            "multiple_slots_for_single_group": "multiple_time_ranges_for_weekday_group",
            "multiple_days_for_time_slots": "multiple_weekdays_for_time_slots",
            "association_inferred": "weekday_time_association_inferred",
            "association_ambiguous": "weekday_time_association_ambiguous",
            "association_unconfirmed": "weekday_time_association_unconfirmed",
        }[association_status]

        clauses = clause_ranges(line)
        clause_by_id = {clause_id: (start, end) for clause_id, (start, end) in enumerate(clauses)}
        for index, slot in enumerate(time_slots):
            slot_days = assigned_days[index]
            summary_start, summary_end = clause_by_id[clause_by_start[slot.start()]]
            slot_weekdays = list(dict.fromkeys(
                day for day_match in slot_days for day in _days_from_match(day_match)
            ))
            if slot_weekdays:
                duplicate_assignment = any(
                    other_index != index
                    and any(anchor is candidate for anchor in slot_days for candidate in assigned_days[other_index])
                    for other_index in range(len(time_slots))
                )
                if duplicate_assignment:
                    slot_days = []
                    slot_weekdays = []
            slot_ambiguities = [*ambiguities, association_ambiguity]
            if not slot_weekdays:
                slot_ambiguities.append("weekday_time_association_unconfirmed")
            slot_normalized = {"date": None, "weekdays": slot_weekdays, "start_time": None, "end_time": None}
            slot_evidence = {"weekdays": [
                {**_evidence(text, line_start + day_match.start(), line_start + day_match.end()),
                 "days": _days_from_match(day_match)}
                for day_match in slot_days
            ]}
            if slot_weekdays and "start_hour" in slot.groupdict():
                parsed, issues = _parse_time_range(slot)
                slot_normalized["start_time"], slot_normalized["end_time"] = parsed["start"], parsed["end"]
                slot_ambiguities.extend(issues)
                evidence_key = "time_range"
            elif slot_weekdays:
                slot_normalized["start_time"], issue = _parse_clock(
                    slot.group("hour"), slot.group("minute"), slot.group("meridiem")
                )
                if issue:
                    slot_ambiguities.append(issue)
                slot_ambiguities.append("end_time_missing")
                evidence_key = "time"
            else:
                evidence_key = "time_range" if "start_hour" in slot.groupdict() else "time"
            slot_evidence.update(_time_evidence(text, line_start, slot, evidence_key))
            events.append(_event(
                "recurring_class_candidate", text,
                line_start + summary_start, line_start + summary_end,
                slot_normalized, slot_evidence, slot_ambiguities,
            ))
        return events

    if "start_hour" in time_slots[0].groupdict():
        time_match = time_slots[0]
        times, issues = _parse_time_range(time_match)
        normalized["start_time"], normalized["end_time"] = times["start"], times["end"]
        ambiguities.extend(issues)
        evidence.update(_time_evidence(text, line_start, time_match, "time_range"))
    else:
        point = time_slots[0]
        normalized["start_time"], issue = _parse_clock(point.group("hour"), point.group("minute"), point.group("meridiem"))
        if issue:
            ambiguities.append(issue)
        ambiguities.append("end_time_missing")
        evidence.update(_time_evidence(text, line_start, point, "time"))
    events.append(_event(
        "recurring_class_candidate", text, line_start, line_end, normalized, evidence, ambiguities,
    ))
    return events


def extract_schedule_candidates(text: str) -> dict[str, Any]:
    """Extract explicit dated events and recurring weekday/time candidates from text."""
    events = []
    for line_start, line_end in _lines(text):
        line = text[line_start:line_end]
        dates = list(DATE_RE.finditer(line))
        if dates:
            days = list(DAY_RE.finditer(line))
            time_slots = _time_slots(line, dates)
            clause_tokens = [*dates, *days, *time_slots]
            clause_by_start = clause_map(line, clause_tokens)
            clauses = clause_ranges(line)
            date_clause_ids = {clause_by_start[date.start()] for date in dates}
            dates_by_clause = {
                clause_id: [date for date in dates if clause_by_start[date.start()] == clause_id]
                for clause_id in date_clause_ids
            }
            event_groups = dated_event_groups(
                line,
                dates,
                clauses,
                clause_by_start,
                DATED_EVENT_CUE_RE,
                DATED_EVENT_ANCHOR_RE,
            )
            for group in event_groups:
                cue = group.cue
                if cue is None:
                    continue
                start = group.start
                while start < cue.end() and line[start].isspace():
                    start += 1
                ambiguities = ["event_date_association_ambiguous"]
                if len(group.date_candidates) > 1:
                    ambiguities.append("multiple_dates_in_clause")
                ambiguities.extend(
                    issue
                    for date in group.date_candidates
                    for issue in _date_value(date)[1]
                )
                field_evidence = {
                    "event_cue": _evidence(text, line_start + cue.start(), line_start + cue.end()),
                }
                clause_time_slots = [
                    slot for slot in time_slots
                    if clause_by_start[slot.start()] == group.clause_id
                ]
                if clause_time_slots:
                    ambiguities.append("date_time_association_ambiguous")
                    field_evidence["times"] = [
                        _time_evidence(text, line_start, slot, "time")["time"]
                        for slot in clause_time_slots
                    ]
                events.append(_serialize_dated_event_candidate(
                    start=start,
                    end=group.end,
                    date_candidates=group.date_candidates,
                    date_association_status="ambiguous",
                    normalized={"date": None, "weekdays": [], "start_time": None, "end_time": None},
                    field_evidence=field_evidence,
                    ambiguities=ambiguities,
                    text=text,
                    line_start=line_start,
                ))
            group_by_date_start = {
                date.start(): group
                for group in event_groups
                for date in group.date_candidates
            }
            event_context = bool(DATED_EVENT_CUE_RE.search(line))
            for date_match in dates:
                event_group = group_by_date_start[date_match.start()]
                if event_group.cue is not None:
                    continue
                clause_id = event_group.clause_id
                clause_dates = dates_by_clause[clause_id]
                event_dates, event_start, event_end = (
                    list(event_group.date_candidates), event_group.start, event_group.end
                )
                clause_start, clause_end = clauses[clause_id]
                schedule_cues_after_date = list(SCHEDULE_CUE_RE.finditer(
                    line, event_dates[-1].end(), clause_end,
                ))
                if schedule_cues_after_date:
                    schedule_cue = schedule_cues_after_date[0]
                    recurring_suffix = line[schedule_cue.start():clause_end]
                    if DAY_RE.search(recurring_suffix) and _time_slots(recurring_suffix, []):
                        events.extend(_recurring_events(
                            text,
                            line_start + schedule_cue.start(),
                            line_start + clause_end,
                        ))
                        event_end = schedule_cue.start()
                        while event_end > event_start and line[event_end - 1] in " \\t,:;–—-":
                            event_end -= 1
                event_cues = list(DATED_EVENT_ANCHOR_RE.finditer(
                    line, clause_start, event_dates[0].start(),
                ))
                last_event_cue = event_cues[-1] if event_cues else None
                if last_event_cue:
                    recurring_end = last_event_cue.start()
                    while recurring_end > clause_start and line[recurring_end - 1] in " \\t,:–—-":
                        recurring_end -= 1
                    recurring_prefix = line[clause_start:recurring_end]
                    if (
                        SCHEDULE_CUE_RE.search(recurring_prefix)
                        and DAY_RE.search(recurring_prefix)
                        and _time_slots(recurring_prefix, [])
                    ):
                        events.extend(_recurring_events(
                            text,
                            line_start + clause_start,
                            line_start + recurring_end,
                        ))
                        event_start = last_event_cue.start()
                event_time_slots = [
                    slot for slot in time_slots
                    if clause_by_start[slot.start()] == clause_id
                    and event_start <= slot.start() < event_end
                ]
                clause_time_slots = [
                    slot for slot in time_slots if clause_by_start[slot.start()] == clause_id
                ]
                event_days = [
                    day for day in days
                    if clause_by_start[day.start()] == clause_id
                    and event_start <= day.start() and day.end() <= event_end
                ]
                date_value, ambiguities = _date_value(date_match)
                if not event_context:
                    ambiguities.append("event_context_unconfirmed")
                if len(event_dates) > 1:
                    ambiguities.append("multiple_dates_in_clause")
                if date_value.get("range_end"):
                    ambiguities.append("event_spans_date_range")
                date_weekdays, weekday_evidence, weekday_issue = _date_weekday_matches(
                    date_match, clause_dates, event_days, clause_by_start,
                )
                if weekday_issue:
                    ambiguities.append(weekday_issue)
                normalized = {
                    "date": date_value,
                    "weekdays": list(dict.fromkeys(
                        day for match in date_weekdays for day in _days_from_match(match)
                    )),
                    "start_time": None,
                    "end_time": None,
                }
                evidence = {"date": _evidence(text, line_start + date_match.start(), line_start + date_match.end())}
                if weekday_evidence:
                    evidence["weekdays"] = [
                        {**_evidence(text, line_start + match.start(), line_start + match.end()), "days": _days_from_match(match)}
                        for match in weekday_evidence
                    ]
                if event_time_slots:
                    if len(event_dates) == 1 and len(event_time_slots) == 1:
                        time_match = event_time_slots[0]
                        times, issues, evidence_key = _time_value(time_match)
                        normalized["start_time"], normalized["end_time"] = times["start"], times["end"]
                        ambiguities.extend(issues)
                        if len(dates) > 1 and len(time_slots) > 1:
                            ambiguities.append("time_date_association_inferred")
                        elif _date_value(date_match)[0].get("range_end"):
                            ambiguities.append("date_range_time_applies_to_start_candidate")
                        evidence.update(_time_evidence(text, line_start, time_match, evidence_key))
                    elif len(event_time_slots) == 1 and len(event_dates) > 1:
                        time_match = event_time_slots[0]
                        explicit_iso_shared_time = (
                            all(item.group("iso_year") for item in event_dates)
                            and re.search(r",\s*$", line[event_dates[-1].end():time_match.start()])
                        )
                        if explicit_iso_shared_time:
                            times, issues, evidence_key = _time_value(time_match)
                            normalized["start_time"], normalized["end_time"] = times["start"], times["end"]
                            ambiguities.extend(issues)
                            ambiguities.append("time_shared_across_multiple_dates")
                            evidence.update(_time_evidence(text, line_start, time_match, evidence_key))
                        else:
                            ambiguities.append("date_time_association_ambiguous")
                            evidence["times"] = [
                                _evidence(text, line_start + time_match.start(), line_start + time_match.end())
                            ]
                    elif len(event_time_slots) == len(event_dates):
                        date_range_flags = [_date_value(item)[0].get("range_end") for item in event_dates]
                        time_match, association_status = date_time_match(
                            date_match, clause_dates, clause_time_slots, clause_by_start,
                        )
                        if association_status == "association_unconfirmed":
                            ambiguities.append("date_time_association_unconfirmed")
                            evidence["times"] = [
                                _evidence(text, line_start + match.start(), line_start + match.end())
                                for match in event_time_slots
                            ]
                        elif association_status == "association_ambiguous":
                            ambiguities.append("date_time_association_ambiguous")
                            evidence["times"] = [
                                _evidence(text, line_start + match.start(), line_start + match.end())
                                for match in event_time_slots
                            ]
                        elif date_range_flags[event_dates.index(date_match)]:
                            ambiguities.append("date_time_association_ambiguous")
                            evidence["times"] = [
                                _evidence(text, line_start + match.start(), line_start + match.end())
                                for match in event_time_slots
                            ]
                        else:
                            times, issues, evidence_key = _time_value(time_match)
                            normalized["start_time"], normalized["end_time"] = times["start"], times["end"]
                            ambiguities.extend(issues)
                            ambiguities.append("time_date_association_inferred")
                            evidence.update(_time_evidence(text, line_start, time_match, evidence_key))
                    else:
                        _, association_status = date_time_match(
                            date_match, clause_dates, clause_time_slots, clause_by_start,
                        )
                        ambiguities.append(
                            "date_time_association_unconfirmed"
                            if association_status == "association_unconfirmed"
                            else "date_time_association_ambiguous"
                        )
                        evidence["times"] = [
                            _evidence(text, line_start + match.start(), line_start + match.end())
                            for match in event_time_slots
                        ]
                elif clause_time_slots and not event_time_slots:
                    ambiguities.append("date_time_association_unconfirmed")
                    evidence["times"] = [
                        _evidence(text, line_start + slot.start(), line_start + slot.end())
                        for slot in clause_time_slots
                    ]
                events.append(_serialize_dated_event_candidate(
                    start=event_start,
                    end=event_end,
                    date_candidates=(date_match,),
                    date_association_status="resolved",
                    normalized=normalized,
                    field_evidence=evidence,
                    ambiguities=ambiguities,
                    text=text,
                    line_start=line_start,
                ))
            for clause_id, (clause_start, clause_end) in enumerate(clauses):
                if clause_id not in date_clause_ids:
                    events.extend(_recurring_events(
                        text,
                        line_start + clause_start,
                        line_start + clause_end,
                    ))
            continue

        events.extend(_recurring_events(text, line_start, line_end))

    events.sort(key=lambda event: event["evidence"]["start"])
    return {
        "schema_version": "pace-schedule/0.1",
        "task": "plain_text_schedule_candidate_extraction",
        "events": events,
        "requires_review": True,
        "commit_policy": "proposals_only_user_confirmation_required",
        "calibration": {
            "status": "uncalibrated",
            "confidence_calibrated": False,
            "reason": "no_independent_human_labeled_schedule_evaluation",
        },
    }
