describe('hotkeys flow', () => {
  it('documents key bindings', () => {
    const bindings = ['ArrowLeft', 'ArrowRight', '1', '2', 'S'];
    expect(bindings).toContain('S');
  });
});
