"""Render the mock academic transcript fixtures from one content specification.

The two fixtures describe the same document in the two layouts a scanner can
produce: a single aligned table, and a row-wise reading of a two-column page
where the left and the right page column share a text line. Keeping both files
generated from one specification means the layouts cannot drift apart, and the
parser can be regression tested against both.

    python scripts/build_mock_transcripts.py
"""
from __future__ import annotations

from pathlib import Path

LEFT_COLUMN_WIDTH = 58
COLUMN_GAP = 3
RIGHT_COLUMN_START = LEFT_COLUMN_WIDTH + COLUMN_GAP

IDENTITY = [
    "ACADEMIC HISTORY",
    "",
    "Pacific High School",
    "509 Lincoln Street",
    "Sitka, AK 99835",
    "Phone: (907) xxx-xxxx",
    "",
    "Lepid, Arthur                           Sex  Male",
    "3160 Harbor Mtn Road                    DOB  4/1/89",
    "Sitka, AK 99835                         SSN",
    "Parent/ Guardian                        Entry Date",
    "                                        Exit Date",
    "                                        Graduation",
    "",
]

TABLE_HEADER = "DATE         GL  TITLE                        MARK  CREDIT"

# (heading, rows) in reading order of the printed page. The first tuple is the
# left page column, the second is the right page column.
LEFT_SECTIONS = (
    ("ELECTIVE", (
        "03/20/04     09  Elective Intensive          A     0.250",
        "03/20/05     10  Elective Intensive          NC    0.000",
        "05/30/05     10  Intro to Programming        B+    0.500",
    )),
    ("LANGUAGE ARTS", (
        "10/30/03     09  CE - Math                   A     0.250",
        "10/30/03     09  Readers Workshop            B+    0.500",
        "12/20/03     09  Content English - Sci       B+    0.250",
        "03/20/04     09  Content English - Math      B     0.250",
        "03/20/04     09  Writers Workshop            A-    0.250",
        "05/30/04     09  Holocaust Literature        A     0.500",
        "05/30/04     09  Revision                    B     0.250",
        "10/30/04     10  Writers Workshop            B-    0.500",
        "12/20/04     10  Content English - Sci       A-    0.250",
        "03/20/05     10  Survival Stories            B+    0.250",
        "12/20/05     11  Content English - SS        B     0.250",
    )),
    ("MATH", (
        "10/30/03     09  Algebra Prep                B+    0.500",
        "03/20/04     09  Algebra 1                   B     0.500",
        "05/30/04     09  Algebra 1                   A-    0.250",
        "10/30/04     10  Algebra 1                   A     0.250",
    )),
    ("PHYSICAL EDUCATION", (
        "05/30/04     09  Independent PE              P     0.750",
        "05/30/05     10  Health                      B+    0.500",
    )),
    ("SCHOOL & COMMUNITY RELATIONS", (
        "10/30/03     09  Orientation                 A     0.250",
        "12/20/03     09  Crew                        B     0.500",
        "05/30/04     09  Crew                        B-    0.250",
        "10/30/04     10  Orientation                 NC    0.000",
        "12/20/04     10  Crew                        B+    0.250",
        "05/30/05     10  Crew                        B     0.250",
        "10/30/05     11  Orientation                 NC    0.000",
        "12/20/05     11  Crew                        A-    0.500",
    )),
    ("SCIENCE", (
        "12/20/03     09  Physical Sci 1              B-    0.500",
        "12/20/04     10  Astronomy                   A-    0.500",
        "10/30/05     11  Life Sci 1                  B+    0.250",
        "12/20/05     11  Human Physiology            B     0.250",
    )),
)

RIGHT_SECTIONS = (
    ("SOCIAL STUDIES", (
        "12/20/03     09  AK Studies                  B     0.500",
        "12/20/04     10  US Government               A-    0.250",
        "03/20/05     10  Global Issues               A     0.250",
        "10/30/05     11  US Government               B     0.500",
        "12/20/05     11  US History                  B-    0.500",
    )),
    ("VOCATIONAL", (
        "12/20/05     11  Work Experience             P     0.750",
    )),
    ("AK HSGQE READING", (
        "04/05/05     10  Reading                     337",
        "  Performance Level: 3 Proficient",
    )),
    ("AK HSGQE WRITING", (
        "04/06/05     10  Writing                     273",
        "  Performance Level: 2 Not Proficient",
    )),
    ("AK HSGQE MATH", (
        "04/07/05     10  Math                        258",
        "  Performance Level: 2 Not Proficient",
    )),
)

FOOTER = (
    "ACADEMIC STANDING",
    "As of:          Total Credits       GPA",
    "12/20/05        13.25",
)

# The right page column ends about two thirds down the printed page.
RIGHT_COLUMN_HEIGHT = 0.62


def _column_lines(sections) -> list[str]:
    lines: list[str] = []
    for heading, rows in sections:
        lines.append(heading)
        lines.extend(rows)
    return lines


def render_aligned() -> str:
    """One continuous table: the left page column followed by the right one."""
    lines = list(IDENTITY)
    lines.append(TABLE_HEADER)
    lines.append("")
    for heading, rows in list(LEFT_SECTIONS) + list(RIGHT_SECTIONS):
        lines.append(heading)
        lines.extend(rows)
        lines.append("")
    lines.extend(FOOTER)
    return "\n".join(lines) + "\n"


def render_rowwise() -> str:
    """A row-wise reading of the two-column page: both columns share a line."""
    left = _column_lines(LEFT_SECTIONS)
    right = _column_lines(RIGHT_SECTIONS)
    slots: list[str | None] = [None] * len(left)
    used: set[int] = set()
    for index, line in enumerate(right):
        position = int(round(RIGHT_COLUMN_HEIGHT * (index + 0.5) / len(right) * len(left))) - 1
        position = max(0, min(len(left) - 1, position))
        while position in used and position + 1 < len(left):
            position += 1
        used.add(position)
        slots[position] = line

    lines = list(IDENTITY)
    header = TABLE_HEADER.ljust(RIGHT_COLUMN_START)
    lines.append(header + TABLE_HEADER)
    for index, left_line in enumerate(left):
        right_line = slots[index]
        if right_line is None:
            lines.append(left_line)
        else:
            lines.append(left_line.ljust(RIGHT_COLUMN_START) + right_line)
    lines.append("")
    lines.extend(FOOTER)
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = root / "data" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "mock_transcript_lepid.txt").write_text(render_aligned(), encoding="utf-8")
    (samples / "mock_transcript_lepid_rowwise.txt").write_text(render_rowwise(), encoding="utf-8")
    print("wrote mock_transcript_lepid.txt and mock_transcript_lepid_rowwise.txt")


if __name__ == "__main__":
    main()
