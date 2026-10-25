import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { SQLiteProvider } from 'expo-sqlite';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { migrate } from '../src/database/schema';
import { ImportProvider } from '../src/features/import/ImportContext';
import { useTheme } from '../src/ui/useTheme';

// The splash image is the app icon's mark on paper; it stays up until React has
// mounted, so the first frame the user sees is the home screen, not a white flash.
SplashScreen.preventAutoHideAsync().catch(() => {});

export default function RootLayout() {
  const { palette, dark } = useTheme();

  useEffect(() => {
    SplashScreen.hideAsync().catch(() => {});
  }, []);

  return (
    <GestureHandlerRootView style={{ flex: 1, backgroundColor: palette.background }}>
      <SafeAreaProvider>
        <StatusBar style={dark ? 'light' : 'dark'} />
        <SQLiteProvider databaseName="pace.db" onInit={migrate}>
          <ImportProvider>
            {/*
              No `animation` overrides anywhere: the native stack's own push, and the
              platform's modal presentation for Scan and Settings, are the transitions
              this app uses. Rebuilding either in JS would be slower and less familiar.
            */}
            <Stack
              screenOptions={{
                headerShown: false,
                contentStyle: { backgroundColor: palette.background },
              }}
            >
              {/* Three destinations live in the native tab bar; everything else
                  is a destination pushed above it or a sheet presented over it. */}
              <Stack.Screen name="(tabs)" />
              <Stack.Screen name="transcript" />
              <Stack.Screen name="academic-record" />
              <Stack.Screen name="schedule/[id]" />
              <Stack.Screen name="compare-pathways" />
              <Stack.Screen name="requirements" />
              <Stack.Screen name="tests" />
              <Stack.Screen name="course/[id]" />
              <Stack.Screen name="editors/course" options={{ presentation: 'modal' }} />
              <Stack.Screen name="editors/year" options={{ presentation: 'modal' }} />
              <Stack.Screen name="editors/schedule" options={{ presentation: 'modal' }} />
              <Stack.Screen name="editors/pathway" options={{ presentation: 'modal' }} />
              <Stack.Screen name="editors/requirement" options={{ presentation: 'modal' }} />
              <Stack.Screen name="editors/test" options={{ presentation: 'modal' }} />
              <Stack.Screen name="targets" options={{ presentation: 'modal' }} />
              <Stack.Screen name="gpa-scale" options={{ presentation: 'modal' }} />
              <Stack.Screen name="scan" options={{ presentation: 'modal' }} />
              <Stack.Screen name="review" />
              <Stack.Screen name="settings" options={{ presentation: 'modal' }} />
              <Stack.Screen name="debug" />
            </Stack>
          </ImportProvider>
        </SQLiteProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
