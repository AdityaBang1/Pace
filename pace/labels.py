"""Stable output labels for the bounded Pace semantic perception model."""

TASK_LABELS = {
    "document": [
        "syllabus", "schedule", "gradebook", "transcript", "report_card",
        "assignment_sheet", "course_catalog", "prerequisite_guide",
        "graduation_requirements", "course_selection", "unknown",
    ],
    "section": [
        "course_overview", "grading", "schedule", "assignments", "prerequisites",
        "learning_outcomes", "policies", "contact", "credits", "unknown",
    ],
    "relation": [
        "prerequisite_of", "corequisite_of", "same_course_as", "weight_for",
        "due_date_for", "no_relation",
    ],
}

ENTITY_LABELS = [
    "O", "B-COURSE", "I-COURSE", "B-COURSE_CODE", "I-COURSE_CODE",
    "B-INSTRUCTOR", "I-INSTRUCTOR", "B-DATE", "I-DATE", "B-GRADE", "I-GRADE",
    "B-CREDIT", "I-CREDIT", "B-ROOM", "I-ROOM", "B-TIME", "I-TIME",
    "B-PERCENT", "I-PERCENT", "B-ASSIGNMENT", "I-ASSIGNMENT", "B-TERM", "I-TERM",
]

IGNORE_INDEX = -100
