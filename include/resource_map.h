#ifndef _RESOURCE_MAP_H_
#define _RESOURCE_MAP_H_

#include <vector>

class ResourceMap
{
public:
  static ResourceMap &getInstance()
  {
    static ResourceMap instance;
    return instance;
  }
  
  ResourceMap(const ResourceMap &) = delete;
  void operator= (const ResourceMap &) = delete;
  
  // Consumes food from a sector and surrounding sectors
  // Returns whether or not enough food was available
  bool consume(const double x, const double y, const int amt);
  void reset();
  
private:
  ResourceMap();
  bool removeFood(const int sliceNum, const int sectorNum, const int amt);
  
  static int coordToSlice(const double x, const double y);
  static int coordToSector(const double x, const double y);
  
  static const int FOOD_FACTOR = 8; // Amount of food in first sector
  static const int NUM_SLICES = 90; // Number of slices per circle
  static const double SECTOR_LENGTH; // Distance before next sector
  static const double RADS_PER_SLICE;
  
  std::vector<int> rMap[NUM_SLICES];
};

#endif