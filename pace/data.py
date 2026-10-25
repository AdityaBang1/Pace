"""Synthetic and provenance-aware data utilities for Pace experiments."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .labels import TASK_LABELS

SEED = 20261019
COURSES = [
    ("BIO 101", "Introduction to Biology"), ("BIO 202", "Cell Biology"),
    ("CHEM 110", "General Chemistry"), ("CHEM 210", "Organic Chemistry"),
    ("CS 101", "Introduction to Computing"), ("CS 201", "Data Structures"),
    ("ENG 102", "Academic Writing"), ("ENG 210", "English Literature"),
    ("HIST 120", "World History"), ("MATH 115", "College Algebra"),
    ("MATH 221", "Calculus I"), ("PHYS 150", "Classical Mechanics"),
    ("PSY 101", "Introduction to Psychology"), ("STAT 200", "Applied Statistics"),
]
INSTRUCTORS = ["Dr. Ada Lovelace", "Prof. Lin Chen", "Morgan Rivera", "Dr. Sam Patel"]
ROOMS = ["Science Hall 204", "North Building 18", "Room B-12", "Online"]
TERMS = ["Fall 2026", "Spring 2027", "Summer 2026", "Autumn 2025"]
# Five distinct surface realizations per class; entire realizations are held out by split.
PARAPHRASES = {
    ("document", "syllabus"): ("Office hours and grading policy", ["course expectations and evaluation", "teaching plan with assessment details", "instructor guidance and grading rules", "course policies and assessment standards", "syllabus information and evaluation methods"]),
    ("document", "schedule"): ("class timetable", ["weekly meeting calendar", "lecture meeting schedule", "class meeting plan", "weekly course timetable", "meeting-time listing"]),
    ("document", "gradebook"): ("Gradebook:", ["Marks ledger:", "Current scores:", "Grade report:", "Assessment marks:", "Student grade record:"]),
    ("document", "transcript"): ("Academic transcript.", ["Official academic record.", "Completed course history.", "Student transcript record.", "Term-by-term academic history.", "Academic achievement record."]),
    ("document", "report_card"): ("Student report card:", ["Term progress report:", "Academic performance summary:", "School marks report:", "Learner results statement:", "Progress card:"]),
    ("document", "assignment_sheet"): ("Lab report due", ["Required laboratory write-up deadline", "Submit the experiment report by", "Lab assignment submission date", "Laboratory task deadline", "Science report submission deadline"]),
    ("document", "course_catalog"): ("Catalog description for", ["Course directory entry for", "Academic catalog listing:", "Program course overview:", "Official course offering:", "Course guide summary for"]),
    ("document", "prerequisite_guide"): ("Prerequisite guide:", ["Enrollment preparation guide:", "Prior-course requirements:", "Eligibility rules:", "Before registration, complete:", "Required prior study:"]),
    ("document", "graduation_requirements"): ("Graduation requirements:", ["Degree completion criteria:", "To earn the diploma, students must:", "Program credit obligations:", "Conditions for graduation:", "Degree audit rules:"]),
    ("document", "course_selection"): ("Course selection form:", ["Registration choice sheet:", "Enrollment planning form:", "Term registration choices:", "Student class selection:", "Course registration request:"]),
    ("section", "course_overview"): ("introduces foundational topics", ["provides an overview of core ideas", "covers essential principles", "presents the main subject concepts", "surveys foundational material", "offers a grounding in key topics"]),
    ("section", "grading"): ("Grading breakdown:", ["Assessment weights:", "Course marks are allocated as follows:", "Evaluation scheme:", "Grade composition:", "Scoring percentages:"]),
    ("section", "schedule"): ("Class meetings:", ["Weekly sessions:", "Meeting pattern:", "Lecture timetable:", "When the class meets:", "Instructional hours:"]),
    ("section", "assignments"): ("Assignments include", ["Students complete", "Required coursework consists of", "The work for this class includes", "Submit the following tasks:", "Course deliverables are"]),
    ("section", "prerequisites"): ("Prerequisite:", ["Required beforehand:", "Prior preparation:", "Enrollment depends on completing", "Students must first pass", "Eligibility prerequisite:"]),
    ("section", "learning_outcomes"): ("By the end,", ["After completing the course,", "Successful students will be able to", "Learning goals include the ability to", "Students should finish able to", "Upon completion, learners can"]),
    ("section", "policies"): ("Attendance policy:", ["Class participation rules:", "Course conduct guidance:", "Absence and late-work rules:", "Student responsibilities:", "Participation expectations:"]),
    ("section", "contact"): ("Instructor:", ["Teaching contact:", "Faculty office hours:", "Course staff contact:", "To reach the professor:", "Office-hour information:"]),
    ("section", "credits"): ("This course carries", ["The class is worth", "Academic credit awarded:", "Students earn", "Credit value:", "The course grants"]),
    ("relation", "prerequisite_of"): ("Completion of", ["Before enrolling in", "Students must first pass", "A prior course requirement for", "Requires successful study of", "Enrollment in requires prior completion of"]),
    ("relation", "corequisite_of"): ("must be taken concurrently with", ["is a concurrent requirement for", "should be enrolled alongside", "is taken in the same term as", "requires simultaneous enrollment in", "must accompany"]),
    ("relation", "same_course_as"): ("Catalog entry", ["Course listing", "The directory names", "Equivalent catalog record:", "This course appears under both names:", "The matching offering is"]),
    ("relation", "weight_for"): ("The final examination is worth 30%", ["Final exam contributes 30 percent", "Examination weighting is 30%", "The final test counts for thirty percent", "30% of marks come from the final", "The final assessment accounts for 30%"]),
    ("relation", "due_date_for"): ("The research project for", ["Submit the course investigation by", "Project submission deadline for", "The final paper in", "Research assignment due for", "Turn in the project in"]),
    ("relation", "no_relation"): ("is listed in the catalog.", ["appears in the course directory.", "is offered by the department.", "has a catalog entry.", "appears among available courses.", "is a listed course."]),
}


def _record(text: str, task: str, label: str, group: str, source: str = "synthetic") -> dict[str, Any]:
    return {"text": text, "task": task, "label": label, "group_id": group,
            "source": source, "provenance": "template-generated" if source == "synthetic" else source}


def generate_examples(count: int = 6000, seed: int = SEED) -> list[dict[str, Any]]:
    """Create controlled examples; OCR/layout families remain intact for split isolation."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    class_counts: dict[tuple[str, str], int] = {}
    for i in range(count):
        code, title = rng.choice(COURSES)
        inst, term, room = rng.choice(INSTRUCTORS), rng.choice(TERMS), rng.choice(ROOMS)
        task = ("document", "section", "relation")[i % 3]
        labels = TASK_LABELS[task]
        label = labels[(i // 3) % len(labels)]
        if task == "document":
            text = {
                "syllabus": f"{code}: {title}. Instructor: {inst}. {term}. Office hours and grading policy.",
                "schedule": f"{term} class timetable: {code} meets Monday and Wednesday, 10:30 AM, {room}.",
                "gradebook": f"Gradebook: {code} {title}. Homework 88%, quizzes 91%, final exam 84%.",
                "transcript": f"Academic transcript. {code} {title}, 3 credits, grade B+. Term: {term}.",
                "report_card": f"Student report card: {title}, final grade A-, credits earned: 3.",
                "assignment_sheet": f"{code} {title}: Lab report due October 22, worth 15 percent of course grade.",
                "course_catalog": f"Catalog description for {code} {title}. Three credit hours. Lecture and lab.",
                "prerequisite_guide": f"Prerequisite guide: {code} requires {rng.choice(COURSES)[0]} with grade C or better.",
                "graduation_requirements": "Graduation requirements: complete 120 credits, including 6 credits in humanities.",
                "course_selection": f"Course selection form: choose {code} {title} for {term}; credits 3.",
                "unknown": f"{rng.choice(['Community garden volunteer signup', 'Water bill statement', 'Bus route map'])}; {rng.randint(1,99)} {rng.choice(['Oak','Main','Pine'])} Street.",
            }[label]
        elif task == "section":
            text = {
                "course_overview": f"{code} {title} introduces foundational topics and includes lectures, labs, and weekly practice.",
                "grading": "Grading breakdown: projects 30%, homework 20%, midterm 20%, final examination 30%.",
                "schedule": f"Class meetings: Mondays and Wednesdays, 10:30 AM to 11:45 AM, {room}.",
                "assignments": "Assignments include weekly problem sets and a final research project due December 12.",
                "prerequisites": f"Prerequisite: {rng.choice(COURSES)[0]} or instructor permission; minimum grade C.",
                "learning_outcomes": "By the end, students will analyze evidence, explain core concepts, and solve problems.",
                "policies": "Attendance policy: notify the instructor about absences; late work loses ten percent per day.",
                "contact": f"Instructor: {inst}. Office hours Tuesday 2 PM in {room}.",
                "credits": "This course carries 3 semester credit hours and meets for four contact hours weekly.",
                "unknown": f"{rng.choice(['Campus dining options', 'Library opening hours', 'Parking permit information'])} are listed online.",
            }[label]
        else:
            parent, candidate = rng.choice(COURSES), rng.choice(COURSES)
            text = {
                "prerequisite_of": f"Completion of {candidate[0]} with a grade of C or higher is required before enrolling in {parent[0]}.",
                "corequisite_of": f"{candidate[0]} must be taken concurrently with {parent[0]}.",
                "same_course_as": f"Catalog entry {parent[0]} {parent[1]} is also listed as {parent[0]}: {parent[1]}.",
                "weight_for": f"The final examination is worth 30% of the course grade in {parent[0]}.",
                "due_date_for": f"The research project for {parent[0]} must be submitted by October 22.",
                "no_relation": f"{parent[0]} is listed in the catalog. Separately, {candidate[0]} meets on Tuesday.",
            }[label]
        # Split by paraphrase family (not random rows); OCR/layout variants stay randomized within each family.
        class_key = (task, label)
        occurrence = class_counts.get(class_key, 0)
        class_counts[class_key] = occurrence + 1
        variant, style = occurrence % 5, (occurrence // 5) % 7
        paraphrase = PARAPHRASES.get(class_key)
        if paraphrase:
            original, variants = paraphrase
            text = text.replace(original, variants[variant])
        if style == 0:
            text = text.replace(" ", "   ")
        elif style == 1:
            text = text.replace(":", " : ").replace("%", " %")
        elif style == 2:
            text = text.replace(" and ", " & ").replace("Course", "C0urse")
        elif style == 3:
            text = text.upper()
        elif style == 4:
            text = text.replace(".", "\n")
        rows.append(_record(text, task, label, f"synthetic:{task}:{label}:{variant}"))
    return rows


def split_by_group(rows: list[dict[str, Any]], seed: int = SEED,
                   fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[dict[str, Any]]]:
    """Split by paraphrase family: 0-2 train, 3 validation, 4 test."""
    if abs(sum(fractions) - 1) > 1e-8 or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to one")
    groups = sorted({r["group_id"] for r in rows})
    if groups and all(g.startswith("synthetic:") for g in groups):
        assignment = {}
        for group in groups:
            variant = int(group.rsplit(":", 1)[1])
            assignment[group] = "train" if variant < 3 else ("validation" if variant == 3 else "test")
    else:
        random.Random(seed).shuffle(groups)
        cut_a = round(len(groups) * fractions[0])
        cut_b = round(len(groups) * (fractions[0] + fractions[1]))
        assignment = {g: "train" for g in groups[:cut_a]}
        assignment.update({g: "validation" for g in groups[cut_a:cut_b]})
        assignment.update({g: "test" for g in groups[cut_b:]})
    return {name: [r for r in rows if assignment[r["group_id"]] == name]
            for name in ("train", "validation", "test")}


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def document_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()
