import React, {useCallback, useRef, useState} from 'react';
import {Box, Text, useInput, usePaste} from 'ink';

type PromptProps = {
  width: number;
  disabled: boolean;
  disabledMessage?: string;
  onSubmit: (value: string) => void;
  onCancel: () => void;
  onExit: () => void;
};

export function Prompt({
  width,
  disabled,
  disabledMessage = 'waiting…',
  onSubmit,
  onCancel,
  onExit,
}: PromptProps) {
  const [value, setValue] = useState('');
  const [cursor, setCursor] = useState(0);
  const valueRef = useRef('');
  const cursorRef = useRef(0);

  const commit = useCallback((nextValue: string, nextCursor: number) => {
    valueRef.current = nextValue;
    cursorRef.current = nextCursor;
    setValue(nextValue);
    setCursor(nextCursor);
  }, []);

  const insert = useCallback((input: string) => {
    const clean = input.replace(/[\r\n]+/g, ' ').replace(/[\u0000-\u001f\u007f]/g, '');
    if (!clean) return;
    const chars = Array.from(valueRef.current);
    const position = cursorRef.current;
    const inserted = Array.from(clean);
    chars.splice(position, 0, ...inserted);
    commit(chars.join(''), position + inserted.length);
  }, [commit]);

  usePaste(insert, {isActive: !disabled});
  useInput((input, key) => {
    if (key.ctrl && input.toLowerCase() === 'c') {
      onExit();
      return;
    }
    if (key.escape) {
      onCancel();
      return;
    }
    if (disabled) return;

    const currentValue = valueRef.current;
    const currentCursor = cursorRef.current;
    const chars = Array.from(currentValue);
    if (key.return) {
      const submitted = currentValue.trim();
      if (submitted) {
        onSubmit(submitted);
        commit('', 0);
      }
    } else if (key.leftArrow) {
      commit(currentValue, Math.max(0, currentCursor - 1));
    } else if (key.rightArrow) {
      commit(currentValue, Math.min(chars.length, currentCursor + 1));
    } else if (key.home) {
      commit(currentValue, 0);
    } else if (key.end) {
      commit(currentValue, chars.length);
    } else if (key.backspace && currentCursor > 0) {
      chars.splice(currentCursor - 1, 1);
      commit(chars.join(''), currentCursor - 1);
    } else if (key.delete && currentCursor < chars.length) {
      chars.splice(currentCursor, 1);
      commit(chars.join(''), currentCursor);
    } else if (!key.ctrl && !key.meta && input) {
      insert(input);
    }
  });

  const available = Math.max(8, width - 7);
  const chars = Array.from(value);
  const start = Math.max(0, cursor - available + 1);
  const visible = chars.slice(start, start + available);
  const localCursor = cursor - start;
  const before = visible.slice(0, localCursor).join('');
  const underCursor = visible[localCursor] ?? ' ';
  const after = visible.slice(localCursor + 1).join('');

  return (
    <Box borderStyle="round" borderColor="gray" paddingX={1} width={width}>
      <Text color="white">{'>'} </Text>
      {disabled ? (
        <Text dimColor>{disabledMessage}</Text>
      ) : (
        <Text wrap="truncate">
          {start > 0 ? '…' : ''}{before}
          <Text inverse>{underCursor}</Text>
          {after}
        </Text>
      )}
    </Box>
  );
}
