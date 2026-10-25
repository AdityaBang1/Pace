import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useState } from 'react';
import { Alert, StyleSheet, Text, TextInput, View } from 'react-native';

import {
  deleteRequirement,
  ensureStructure,
  readStructure,
  saveRequirement,
  setCourseRequirements,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import { listCourses, type CourseRow } from '../../src/database/repo';
import { statusOf } from '../../src/domain/record';
import { Button, Chip, Notice, PageHeader, Screen, SectionHeader } from '../../src/ui/components';
import { hairline, radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

/**
 * One requirement: a name, a credit goal, and the courses that count towards it.
 *
 * Mapping is explicit. Pace never decides that "AP Human Geography" is a Social
 * Studies credit because a pattern in its title said so; the student taps it.
 */
export default function RequirementEditor() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [originalLinks, setOriginalLinks] = useState<Record<string, string[]>>({});
  const [name, setName] = useState('');
  const [credits, setCredits] = useState('');
  const [selected, setSelected] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      await ensureStructure(db);
      const [next, rows] = await Promise.all([readStructure(db), listCourses(db)]);
      setStructure(next);
      setCourses(rows);
      setOriginalLinks(next.links);
      if (id) {
        const requirement = next.requirements.find((row) => row.id === id);
        if (requirement) {
          setName(requirement.name);
          setCredits(String(requirement.required_credits));
        }
        setSelected(next.links[id] ?? []);
      }
    })();
  }, [db, id]);

  const value = Number(credits.replace(',', '.'));
  const valid = name.trim().length > 0 && Number.isFinite(value) && value > 0;

  const toggle = (courseId: string) => {
    setSelected((current) =>
      current.includes(courseId) ? current.filter((row) => row !== courseId) : [...current, courseId],
    );
  };

  const save = async () => {
    if (!valid) return;
    setSaving(true);
    try {
      const requirementId = await saveRequirement(db, {
        id,
        name: name.trim(),
        requiredCredits: Math.round(value * 100) / 100,
      });
      // Every course whose membership changed is written, so a course can belong to
      // more than one requirement without either edit erasing the other.
      for (const course of courses) {
        const previous = (originalLinks[requirementId] ?? []).includes(course.id);
        const now = selected.includes(course.id);
        if (previous === now) continue;
        const others = Object.entries(originalLinks)
          .filter(([requirement]) => requirement !== requirementId)
          .filter(([, ids]) => ids.includes(course.id))
          .map(([requirement]) => requirement);
        await setCourseRequirements(db, course.id, now ? [...others, requirementId] : others);
      }
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.back();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The requirement could not be saved.');
      setSaving(false);
    }
  };

  const remove = () => {
    if (!id) return;
    Alert.alert('Delete requirement?', name || 'This requirement', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await deleteRequirement(db, id);
            router.back();
          })();
        },
      },
    ]);
  };

  const grouped = (structure?.years ?? [])
    .map((year) => ({
      year,
      rows: courses.filter(
        (course) => course.academic_year_id === year.id && statusOf(course) !== 'planned',
      ),
    }))
    .filter((group) => group.rows.length > 0);

  return (
    <Screen scroll>
      <PageHeader title={id ? 'Edit Requirement' : 'New Requirement'} onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
          <Notice tone="danger" body={error} />
        </View>
      ) : null}

      <View style={{ paddingHorizontal: space.xl, gap: space.md, paddingTop: space.sm }}>
        <View>
          <Text style={[type.caption, { color: palette.inkFaint, marginBottom: 4 }]}>Name</Text>
          <TextInput
            value={name}
            onChangeText={setName}
            placeholder="Math"
            placeholderTextColor={palette.inkFaint}
            style={[
              styles.input,
              type.body,
              { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface },
            ]}
          />
        </View>
        <View>
          <Text style={[type.caption, { color: palette.inkFaint, marginBottom: 4 }]}>Credits required</Text>
          <TextInput
            value={credits}
            onChangeText={setCredits}
            placeholder="4.0"
            keyboardType="decimal-pad"
            placeholderTextColor={palette.inkFaint}
            style={[
              styles.input,
              type.number,
              { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface },
            ]}
          />
        </View>
      </View>

      <SectionHeader>COURSES</SectionHeader>
      {grouped.map((group) => (
        <View key={group.year.id} style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
          <Text style={[type.caption, { color: palette.inkFaint, marginBottom: space.xs }]} numberOfLines={1}>
            {group.year.name}
          </Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
            {group.rows.map((course) => (
              <Chip
                key={course.id}
                label={course.title ?? course.code ?? 'Course'}
                selected={selected.includes(course.id)}
                onPress={() => toggle(course.id)}
              />
            ))}
          </View>
        </View>
      ))}
      {!grouped.length ? (
        <View style={{ paddingHorizontal: space.xl }}>
          <Text style={[type.caption, { color: palette.inkMuted }]}>No courses to map yet.</Text>
        </View>
      ) : null}

      <View style={{ paddingHorizontal: space.xl, paddingTop: space.lg, gap: space.md }}>
        <Button title="Save" onPress={() => void save()} disabled={!valid} loading={saving} />
        {id ? <Button title="Delete" kind="quiet" onPress={remove} /> : null}
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  input: {
    minHeight: 44,
    paddingHorizontal: space.md,
    paddingVertical: space.sm,
    borderRadius: radius.md,
    borderWidth: hairline,
  },
});
