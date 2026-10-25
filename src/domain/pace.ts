/**
 * Pace: is the chosen path actually reaching the targets the student set?
 *
 * Deterministic by construction. There is no score, no prediction and no prose
 * generated from a model: it compares a projection with numbers the user typed and
 * reports the differences. With no targets set it says the setup is incomplete
 * rather than inventing a verdict.
 */
import type { RequirementProgress } from './requirements';
import type { PathwayView } from './pathways';

export interface Targets {
  gpa: number | null;
  credits: number | null;
}

export interface PaceIssue {
  type: 'gpa' | 'credits' | 'requirement';
  label: string;
  /** The gap, always positive: how much is missing. */
  missing: number;
}

export interface PaceStatus {
  state: 'on_pace' | 'needs_attention' | 'incomplete_setup';
  issues: PaceIssue[];
  /** Which pathway the verdict is about, for the label on Home. */
  pathwayName: string | null;
}

const round = (value: number) => Math.round(value * 100) / 100;

export function evaluatePace(
  pathway: PathwayView | null,
  requirements: RequirementProgress[],
  targets: Targets,
): PaceStatus {
  if (!pathway) return { state: 'incomplete_setup', issues: [], pathwayName: null };
  const issues: PaceIssue[] = [];
  if (targets.gpa != null) {
    if (pathway.gpa.weighted == null) {
      issues.push({ type: 'gpa', label: 'Needs grades for a GPA', missing: 0 });
    } else if (pathway.gpa.weighted < targets.gpa) {
      issues.push({
        type: 'gpa',
        label: `Projected GPA ${round(targets.gpa - pathway.gpa.weighted)} below target`,
        missing: round(targets.gpa - pathway.gpa.weighted),
      });
    }
  }
  if (targets.credits != null) {
    const planned = round(pathway.credits.earned + pathway.credits.inProgress + pathway.credits.planned);
    if (planned < targets.credits) {
      const missing = round(targets.credits - planned);
      issues.push({ type: 'credits', label: `Needs ${missing} credits`, missing });
    }
  }
  for (const progress of requirements) {
    if (progress.complete) continue;
    const reachable =
      progress.requirement.required_credits - progress.earned - progress.inProgress - progress.planned;
    if (reachable > 0) {
      issues.push({
        type: 'requirement',
        label: `Needs ${round(reachable)} ${progress.requirement.name} credits`,
        missing: round(reachable),
      });
    }
  }
  const hasTarget = targets.gpa != null || targets.credits != null || requirements.length > 0;
  if (!hasTarget) return { state: 'incomplete_setup', issues: [], pathwayName: pathway.pathway.name };
  return {
    state: issues.length ? 'needs_attention' : 'on_pace',
    issues,
    pathwayName: pathway.pathway.name,
  };
}

/** Targets live in `settings`; a blank or unreadable value means "not set". */
export function parseTargets(json: { gpa: string; credits: string }): Targets {
  const number = (raw: string): number | null => {
    const value = Number(raw);
    return raw.trim() !== '' && Number.isFinite(value) ? value : null;
  };
  return { gpa: number(json.gpa), credits: number(json.credits) };
}

export const GPA_TARGET_KEY = 'target_gpa';
export const CREDIT_TARGET_KEY = 'target_credits';
