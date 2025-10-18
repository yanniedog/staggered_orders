from usage_tracker import UsageTracker
import os

# Clean up test file first
if os.path.exists('test.json'):
    os.remove('test.json')

# Test usage tracking
tracker = UsageTracker('test.json')
print('Initial usage stats:', len(tracker.get_stats()))

# Track some usage
tracker.track_usage(3, 20, 720, 1000, 'kelly_optimized', 'SOLUSDT', 'linear')
print('After tracking:', len(tracker.get_stats()))

# Check if file exists before save
print('File exists before save:', os.path.exists('test.json'))

# Save stats
tracker.save_stats()
print('File exists after save:', os.path.exists('test.json'))

# Clean up
if os.path.exists('test.json'):
    os.remove('test.json')
    print('Test completed successfully')

