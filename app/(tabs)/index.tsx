import {
  Button as NativeButton,
  HStack,
  Section,
  Spacer,
  Text as NativeText,
  VStack,
} from '@expo/ui/swift-ui';
import { font, foregroundStyle, monospacedDigit, padding } from '@expo/ui/swift-ui/modifiers';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';

import {
  ensureDefaults,
  ensureStructure,
  readStructure,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import {
  getSetting,
  latestTranscript,
  listCourses,
  readGpaScaleValue,
  type TranscriptRow,
} from '../../src/database/repo';
import { creditsValue, gpaValue } from '../../src/domain/academics';
import { parseScale } from '../../src/domain/gpa';
import {
  CREDIT_TARGET_KEY,
  GPA_TARGET_KEY,
  evaluatePace,
  parseTargets,
  type PaceStatus,
} from '../../src/domain/pace';
import { buildPathway, type PathwayView } from '../../src/domain/pathways';
import { buildRecord, type RecordView } from '../../src/domain/record';
import { requirementProgress } from '../../src/domain/requirements';
import { HeaderPill, LabeledContent, NativeScreen, Value } from '../../src/ui/native';

/**
 * Home answers one question: where am I academically right now.
 *
 * Drawn as a system form, like the rest of the reading screens: the headline number
 * in a section of its own, then credits, the courses being taken, and the plan the
 * rest of the app is measuring. Every row that leaves Home is a native button, so
 * the whole screen has the platform's press behaviour rather than a bespoke one.
 */
interface HomeState {
  record: RecordView;
  transcript: TranscriptRow | null;
  pace: PaceStatus;
  pathway: PathwayView | null;
  hasTargets: boolean;
  /** A value the page printed is never presented as one Pace derived. */
  gpa: { value: number; label: string } | null;
}

const quiet = foregroundStyle({ type: 'hierarchical', style: 'secondary' });

export default function Home() {
  const db = useSQLiteContext();
  const router = useRouter();
  const [data, setData] = useState<HomeState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      // A course imported before the academic structure existed has no year or
      // schedule; placing it first means every number below has a place to come from.
      await ensureStructure(db);
      await ensureDefaults(db);
      const [courses, transcript, scaleJson, gpaTarget, creditTarget] = await Promise.all([
        listCourses(db),
        latestTranscript(db),
        readGpaScaleValue(db),
        getSetting(db, GPA_TARGET_KEY, ''),
        getSetting(db, CREDIT_TARGET_KEY, ''),
      ]);
      const structure: AcademicStructure = await readStructure(db);
      const scale = parseScale(scaleJson);
      const record = buildRecord(scale, structure.years, structure.terms, structure.schedules, courses);
      const active =
        structure.pathways.find((row) => row.is_active === 1) ?? structure.pathways[0] ?? null;
      const pathway = active
        ? buildPathway(active, scale, structure.years, structure.schedules, courses)
        : null;
      const targets = parseTargets({ gpa: gpaTarget, credits: creditTarget });
      const pace = evaluatePace(
        pathway,
        requirementProgress(structure.requirements, structure.links, courses),
        targets,
      );
      const calculated = record.gpa.weighted;
      const reported = transcript?.reported_gpa ?? null;
      const gpa =
        calculated != null
          ? { value: calculated, label: 'Weighted · Calculated' }
          : reported != null
            ? { value: reported, label: 'Weighted · Reported' }
            : null;
      setData({
        record,
        transcript,
        pace,
        pathway,
        hasTargets: targets.gpa != null || targets.credits != null || structure.requirements.length > 0,
        gpa,
      });
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    }
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      let live = true;
      void (async () => {
        if (!live) return;
        await refresh();
      })();
      return () => {
        live = false;
      };
    }, [refresh]),
  );

  const current = data?.record.current.slice(0, 5) ?? [];
  const issue = data?.pace.issues[0] ?? null;
  const showVerdict = data != null && data.pace.state !== 'incomplete_setup';

  return (
    <NativeScreen
      title="Pace"
      right={
        <HeaderPill
          actions={[
            { systemName: 'doc.badge.plus', label: 'Import a transcript', onPress: () => router.push('/scan') },
            { systemName: 'gearshape.fill', label: 'Settings', onPress: () => router.push('/settings') },
          ]}
        />
      }
    >
      {error ? (
        <Section title="Your data could not be loaded">
          <NativeText>{error}</NativeText>
          <NativeButton
            label="Try Again"
            systemImage="arrow.clockwise"
            onPress={() => void refresh()}
          />
        </Section>
      ) : null}

      {!error && data && data.record.courseCount === 0 ? (
        <Section title="No academics yet">
          <NativeText>Import a transcript to begin.</NativeText>
          <NativeButton
            label="Import Transcript"
            systemImage="doc.badge.plus"
            onPress={() => router.push('/scan')}
          />
        </Section>
      ) : null}

      {!error && data && data.record.courseCount > 0 ? (
        <>
          {showVerdict ? (
            <Section
              title={issue ? 'Needs attention' : 'On pace'}
              footer={
                <NativeText>
                  {issue ? issue.label : `${data.pace.pathwayName ?? 'Your plan'} reaches your targets`}
                </NativeText>
              }
            >
              <NativeButton
                label={issue ? 'Requirements' : 'View plan'}
                systemImage={issue ? 'exclamationmark.triangle' : 'checkmark.circle'}
                onPress={() => router.push(issue ? '/requirements' : '/plan')}
              />
            </Section>
          ) : null}

          {data.gpa ? (
            <Section>
              <VStack
                alignment="leading"
                spacing={2}
                modifiers={[padding({ horizontal: 16, vertical: 8 })]}
              >
                <NativeText
                  modifiers={[
                    font({ textStyle: 'largeTitle', weight: 'bold', design: 'rounded' }),
                    monospacedDigit(),
                  ]}
                >
                  {gpaValue(data.gpa.value)}
                </NativeText>
                <NativeText modifiers={[quiet]}>{data.gpa.label}</NativeText>
              </VStack>
            </Section>
          ) : null}

          <Section title="Credits">
            <HStack spacing={24} modifiers={[padding({ vertical: 4 })]}>
              <VStack alignment="leading" spacing={2}>
                <NativeText
                  modifiers={[font({ textStyle: 'title2', weight: 'semibold' }), monospacedDigit()]}
                >
                  {creditsValue(data.record.credits.earned)}
                </NativeText>
                <NativeText modifiers={[quiet]}>Earned</NativeText>
              </VStack>
              <VStack alignment="leading" spacing={2}>
                <NativeText
                  modifiers={[font({ textStyle: 'title2', weight: 'semibold' }), monospacedDigit()]}
                >
                  {creditsValue(data.record.credits.inProgress)}
                </NativeText>
                <NativeText modifiers={[quiet]}>In Progress</NativeText>
              </VStack>
            </HStack>
            {!data.hasTargets ? (
              <NativeButton
                label="Set Targets"
                systemImage="target"
                onPress={() => router.push('/targets')}
              />
            ) : null}
          </Section>

          {current.length ? (
            <Section
              title="Current"
              footer={<NativeText>{`${data.record.current.length} in progress`}</NativeText>}
            >
              {current.map((course) => (
                <NativeButton
                  key={course.id}
                  onPress={() => router.push(`/course/${course.id}`)}
                >
                  <HStack spacing={8}>
                    <NativeText>{course.title ?? course.code ?? 'Course'}</NativeText>
                    <Spacer />
                    <NativeText modifiers={[quiet]}>{creditsValue(course.credits)}</NativeText>
                  </HStack>
                </NativeButton>
              ))}
              <NativeButton
                label="See All"
                systemImage="chevron.right"
                onPress={() => router.push('/academics')}
              />
            </Section>
          ) : null}

          {data.pathway ? (
            <Section
              title="Plan"
              footer={
                <NativeText>
                  {[
                    data.pathway.gpa.weighted != null
                      ? `${gpaValue(data.pathway.gpa.weighted)} projected`
                      : null,
                    `${creditsValue(data.pathway.credits.all)} credits`,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
                </NativeText>
              }
            >
              <NativeButton
                label={data.pathway.pathway.name}
                systemImage="map"
                onPress={() => router.push('/plan')}
              />
            </Section>
          ) : null}

          {data.transcript ? (
            <Section>
              <LabeledContent label="Transcript">
                <Value>{data.transcript.student_name ?? 'Imported'}</Value>
              </LabeledContent>
              <NativeButton
                label="Open Transcript"
                systemImage="doc.text"
                onPress={() => router.push('/transcript')}
              />
            </Section>
          ) : null}
        </>
      ) : null}
    </NativeScreen>
  );
}
