#include "resource_map.h"

#include <math.h>

const int ResourceMap::FOOD_FACTOR;
const int ResourceMap::NUM_SLICES;
const double ResourceMap::RADS_PER_SLICE = 2 * M_PI / NUM_SLICES;
const double ResourceMap::SECTOR_LENGTH = 0.5;

ResourceMap::ResourceMap()
{

}

// Removes food from the point's sector and its 8 surrounding sectors
// Returns whether there was enough food
bool ResourceMap::consume(const double x, const double y)
{
  const int sliceNum = coordToSlice(x, y);
  const int sectorNum = coordToSector(x, y);
  
  const int leftSliceNum = (sliceNum - 1 + NUM_SLICES) % NUM_SLICES;
  const int rightSliceNum = (sliceNum + 1) % NUM_SLICES;
  
  bool enoughFood = true;
  
  // Remove food from top sectors
  enoughFood &= this->removeFood(leftSliceNum, sectorNum + 1, 1);
  enoughFood &= this->removeFood(sliceNum, sectorNum + 1, 1);
  enoughFood &= this->removeFood(rightSliceNum, sectorNum + 1, 1);
  
  // Remove food from middle sectors
  enoughFood &= this->removeFood(leftSliceNum, sectorNum, 1);
  enoughFood &= this->removeFood(sliceNum, sectorNum, 2);
  enoughFood &= this->removeFood(rightSliceNum, sectorNum, 1);
  
  // Remove food from bottom sectors (if exist)
  if(sectorNum > 0)
  {
    enoughFood &= this->removeFood(leftSliceNum, sectorNum - 1, 1);
    enoughFood &= this->removeFood(sliceNum, sectorNum - 1, 1);
    enoughFood &= this->removeFood(rightSliceNum, sectorNum - 1, 1);
  }
  
  return enoughFood;
}

// Resets the map to full food
void ResourceMap::reset()
{
  for(int slice = 0; slice < NUM_SLICES; ++slice)
  {
    int food = 0;
    for(int sect = 0; sect < rMap[slice].size(); ++sect)
    {
      food += FOOD_FACTOR;
      rMap[slice][sect] = food;
    }
  }
}

// Removes food from only the given sector
// Returns whether there was enough food
bool ResourceMap::removeFood(const int sliceNum, const int sectorNum, const int amt)
{
  // Fill in missing sectors with food
  while(rMap[sliceNum].size() <= sectorNum)
  {
    const int food = (rMap[sliceNum].size() + 1) * FOOD_FACTOR;
    rMap[sliceNum].push_back(food);
  }
  
  // If there isn't enough food, remove remaining and fail
  if(rMap[sliceNum][sectorNum] < amt)
  {
    rMap[sliceNum][sectorNum] = 0;
    return false;
  }
  
  // Otherwise, remove amount and succeed
  rMap[sliceNum][sectorNum] -= amt;
  return true;
}

int ResourceMap::coordToSlice(const double x, const double y)
{
  const double angleRads = atan2(y, x);
  
  return (angleRads + M_PI) / RADS_PER_SLICE;
}

int ResourceMap::coordToSector(const double x, const double y)
{
  const double dist = sqrt(x * x + y * y);
  
  return dist / SECTOR_LENGTH;
}