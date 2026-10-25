/**
 * Editing one proposed value.
 *
 * The sheet has exactly one job: let the reviewer correct what the page actually
 * says. Validation is inline and specific — a weight has to be a number between 0
 * and 100, a time has to look like a time — because the alternative is an "Import"
 * button that fails at the database with a message nobody can act on.
 */
import { useEffect, useState } from 'react';
import { Modal, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Button, Chip } from '../../ui/components';
import { radius, space, type } from '../../ui/theme';
import { useTheme } from '../../ui/useTheme';

export type EditTarget =
  | { kind: 'text'; key: string; title: string; value: string; maxLength?: number }
  | { kind: 'number'; key: string; title: string; value: string }
  | { kind: 'weight'; key: string; title: string; category: string; weight: string }
  | {
      kind: 'meeting';
      key: string;
      title: string;
      days: string[];
      startTime: string;
      endTime: string;
      room: string;
    };

const CLOCK = /^([01]?\d|2[0-3]):([0-5]\d)$/;

/** iOS scheduling convention, not the engine's token order. */
const DAYS: { code: string; label: string }[] = [
  { code: 'MO', label: 'Mon' },
  { code: 'TU', label: 'Tue' },
  { code: 'WE', label: 'Wed' },
  { code: 'TH', label: 'Thu' },
  { code: 'FR', label: 'Fri' },
  { code: 'SA', label: 'Sat' },
  { code: 'SU', label: 'Sun' },
];

function clockError(value: string, label: string): string | null {
  if (!value.trim()) return null;
  return CLOCK.test(value.trim()) ? null : `${label} should look like 11:00 (24-hour).`;
}

/** Minutes since midnight, or null when the text is not a clock time. */
function clockMinutes(value: string): number | null {
  const match = CLOCK.exec(value.trim());
  if (!match) return null;
  return Number(match[1]) * 60 + Number(match[2]);
}

const DAY_ORDER = DAYS.map((day) => day.code);

export function EditSheet({
  target,
  visible,
  included,
  onClose,
  onSave,
  onToggleIncluded,
  onViewSource,
  canToggleIncluded = true,
  canViewSource = true,
}: {
  target: EditTarget | null;
  visible: boolean;
  included: boolean;
  onClose: () => void;
  onSave: (target: EditTarget) => void;
  onToggleIncluded: () => void;
  onViewSource: () => void;
  /** A value the reviewer is about to add has nothing to toggle yet. */
  canToggleIncluded?: boolean;
  /** A typed value has no evidence line to show. */
  canViewSource?: boolean;
}) {
  const { palette } = useTheme();
  const insets = useSafeAreaInsets();

  const [text, setText] = useState('');
  const [category, setCategory] = useState('');
  const [weight, setWeight] = useState('');
  const [days, setDays] = useState<string[]>([]);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [room, setRoom] = useState('');

  useEffect(() => {
    if (!target) return;
    setText(target.kind === 'text' || target.kind === 'number' ? target.value : '');
    setCategory(target.kind === 'weight' ? target.category : '');
    setWeight(target.kind === 'weight' ? String(target.weight) : '');
    setDays(target.kind === 'meeting' ? target.days : []);
    setStart(target.kind === 'meeting' ? target.startTime ?? '' : '');
    setEnd(target.kind === 'meeting' ? target.endTime ?? '' : '');
    setRoom(target.kind === 'meeting' ? target.room ?? '' : '');
  }, [target]);

  if (!target) return null;

  const weightNumber = Number(weight.replace('%', '').trim());
  const weightValid = Number.isFinite(weightNumber) && weightNumber >= 0 && weightNumber <= 100;
  const numberValue = Number(text.trim());
  const numberValid = text.trim().length > 0 && Number.isFinite(numberValue) && numberValue >= 0 && numberValue <= 30;
  const clockIssue = target.kind === 'meeting' ? clockError(start, 'Start') ?? clockError(end, 'End') : null;
  // A meeting with no days is not a meeting, and a class that ends before it
  // starts would be stored as written and shown as nonsense later. Both are
  // checked here so the Import button is never the first thing to object.
  const startMinutes = clockMinutes(start);
  const endMinutes = clockMinutes(end);
  const orderError =
    startMinutes != null && endMinutes != null && endMinutes <= startMinutes
      ? 'The end time has to be after the start time.'
      : null;
  const daysError = target.kind === 'meeting' && days.length === 0 ? 'Choose at least one day.' : null;
  const timeError = clockIssue ?? orderError;
  const canSave =
    target.kind === 'text'
      ? text.trim().length > 0
      : target.kind === 'number'
        ? numberValid
        : target.kind === 'weight'
          ? weightValid && category.trim().length > 0
          : timeError == null && daysError == null;

  const save = () => {
    if (!canSave) return;
    if (target.kind === 'text') {
      onSave({ ...target, value: text.trim() });
    } else if (target.kind === 'number') {
      onSave({ ...target, value: String(Math.round(numberValue * 100) / 100) });
    } else if (target.kind === 'weight') {
      onSave({
        ...target,
        category: category.trim(),
        weight: String(Math.round(weightNumber * 10) / 10),
      });
    } else {
      onSave({
        ...target,
        // Taps arrive in tap order; a timetable reads Monday-first.
        days: [...days].sort((left, right) => DAY_ORDER.indexOf(left) - DAY_ORDER.indexOf(right)),
        startTime: start.trim(),
        endTime: end.trim(),
        room: room.trim(),
      });
    }
  };

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={{ flex: 1, backgroundColor: palette.background }}>
        <View style={[styles.header, { borderBottomColor: palette.line }]}>
          <Pressable accessibilityRole="button" onPress={onClose} hitSlop={12} style={{ padding: space.sm }}>
            <Text style={[type.caption, { color: palette.inkMuted }]}>Cancel</Text>
          </Pressable>
          <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={1}>
            {target.title}
          </Text>
          <Pressable accessibilityRole="button" onPress={save} disabled={!canSave} hitSlop={12} style={{ padding: space.sm }}>
            <Text style={[type.caption, { color: canSave ? palette.accent : palette.inkFaint, fontWeight: '600' }]}>
              Save
            </Text>
          </Pressable>
        </View>

        <ScrollView contentContainerStyle={{ padding: space.xl, gap: space.lg }} keyboardShouldPersistTaps="handled">
          {target.kind === 'text' ? (
            <View>
              <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>{target.title}</Text>
              <TextInput
                value={text}
                onChangeText={setText}
                autoFocus
                maxLength={target.maxLength ?? 120}
                style={[styles.input, { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface }]}
                accessibilityLabel={target.title}
              />
            </View>
          ) : null}

          {target.kind === 'number' ? (
            <View>
              <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>{target.title}</Text>
              <TextInput
                value={text}
                onChangeText={setText}
                keyboardType="decimal-pad"
                autoFocus
                style={[styles.input, { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface }]}
                accessibilityLabel={target.title}
              />
              {!numberValid ? (
                <Text style={[type.caption, { color: palette.danger, marginTop: 6 }]}>
                  Enter a number between 0 and 30.
                </Text>
              ) : null}
            </View>
          ) : null}

          {target.kind === 'weight' ? (
            <>
              <View>
                <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>Category</Text>
                <TextInput
                  value={category}
                  onChangeText={setCategory}
                  maxLength={40}
                  style={[styles.input, { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface }]}
                  accessibilityLabel="Category"
                />
              </View>
              <View>
                <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>Weight</Text>
                <TextInput
                  value={weight}
                  onChangeText={setWeight}
                  keyboardType="decimal-pad"
                  style={[styles.input, { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface }]}
                  accessibilityLabel="Weight in percent"
                />
                {!weightValid ? (
                  <Text style={[type.caption, { color: palette.danger, marginTop: 6 }]}>
                    A weight has to be a number between 0 and 100.
                  </Text>
                ) : null}
              </View>
            </>
          ) : null}

          {target.kind === 'meeting' ? (
            <>
              <View>
                <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>Days</Text>
                <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
                  {DAYS.map(({ code, label }) => (
                    <Chip
                      key={code}
                      label={label}
                      selected={days.includes(code)}
                      tone={days.includes(code) ? 'accent' : 'neutral'}
                      onPress={() =>
                        setDays((current) =>
                          current.includes(code)
                            ? current.filter((value) => value !== code)
                            : [...current, code],
                        )
                      }
                    />
                  ))}
                </View>
                {daysError ? (
                  <Text style={[type.caption, { color: palette.danger, marginTop: 6 }]}>{daysError}</Text>
                ) : null}
              </View>
              <View style={{ flexDirection: 'row', gap: space.md }}>
                <View style={{ flex: 1 }}>
                  <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>Starts</Text>
                  <TextInput
                    value={start}
                    onChangeText={setStart}
                    placeholder="11:00"
                    placeholderTextColor={palette.inkFaint}
                    style={[styles.input, { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface }]}
                    accessibilityLabel="Start time"
                  />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>Ends</Text>
                  <TextInput
                    value={end}
                    onChangeText={setEnd}
                    placeholder="12:15"
                    placeholderTextColor={palette.inkFaint}
                    style={[styles.input, { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface }]}
                    accessibilityLabel="End time"
                  />
                </View>
              </View>
              {timeError ? (
                <Text style={[type.caption, { color: palette.danger }]}>{timeError}</Text>
              ) : null}
              <View>
                <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>Room</Text>
                <TextInput
                  value={room}
                  onChangeText={setRoom}
                  maxLength={40}
                  placeholder="Room 118"
                  placeholderTextColor={palette.inkFaint}
                  style={[styles.input, { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface }]}
                  accessibilityLabel="Room"
                />
              </View>
            </>
          ) : null}

          {canViewSource || canToggleIncluded ? (
            <View style={{ gap: space.sm, marginTop: space.sm }}>
              {canViewSource ? <Button title="View source" kind="secondary" onPress={onViewSource} /> : null}
              {canToggleIncluded ? (
                <Button
                  title={included ? 'Exclude from import' : 'Include in import'}
                  kind="quiet"
                  onPress={onToggleIncluded}
                />
              ) : null}
              {!canViewSource ? (
                <Text style={[type.caption, { color: palette.inkFaint }]}>
                  Typed by you, so there is no line on the page to show.
                </Text>
              ) : null}
            </View>
          ) : null}
        </ScrollView>

        <View style={{ padding: space.xl, paddingBottom: insets.bottom + space.lg }}>
          <Button title="Save changes" onPress={save} disabled={!canSave} />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: space.md,
    paddingVertical: space.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  input: {
    borderWidth: 1,
    borderRadius: radius.md,
    paddingHorizontal: space.md,
    paddingVertical: space.md,
    fontSize: 17,
  },
});
